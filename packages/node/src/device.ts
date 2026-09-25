import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { promisify } from 'node:util';
import { ensureBinary, type OnProgress } from './compile.js';
import {
  AbortError,
  ContextLengthError,
  ModelBusyError,
  ModelUnavailableError,
  QuotaError,
  RefusalError,
  SchemaRejectedError,
  AppleLLMError,
  TimeoutError,
  ToolExecutionError,
  UnsupportedError,
  toUnavailableReason,
  unavailableMessage,
} from './errors.js';
import type { HistoryEntry } from './messages.js';
import { ReplayBook } from './messages.js';
import { HelperServer, abortErrorFrom } from './protocol.js';
import { toAppleSchema, type JsonSchema } from './schema.js';
import { describeIssues, restoreNulls } from './standard-schema.js';
import { isAppleSiliconMac, macosMajor } from './target.js';
import { BUILT_IN_TOOLS, toolOutputText, type BuiltInTool, type NormalizedTools } from './tools.js';

export type { BuiltInTool } from './tools.js';

const execFileAsync = promisify(execFile);

/**
 * Sampling temperature. Not zero on purpose.
 *
 * Constrained decoding already guarantees the schema, so greedy decoding buys
 * nothing and reliably degenerates: at 0 the model padded an unbounded array
 * forever, then ran away *inside a single string*, emitting 2.7KB of
 * "tasks-tasks-tasks-…". A 2s call became 20s. Apple honours `maxItems` but
 * ignores `maxLength`, so bounding strings is not available as a fix — a little
 * sampling is. Nothing about this is a quality/creativity tradeoff.
 */
export const DEFAULT_TEMPERATURE = 0.4;

/**
 * Cap on generated tokens. Left uncapped, the model occasionally runs away and
 * only stops when it exhausts the context window — one observed run burned
 * ~206s before failing. This bounds that to well under a minute while leaving
 * room for a few paragraphs of prose.
 */
export const DEFAULT_MAX_TOKENS = 2048;

/**
 * Tool calls allowed in one request before the model is told to stop calling
 * and answer. A small model can loop on a tool whose output it misreads.
 */
export const DEFAULT_MAX_TOOL_CALLS = 8;

/** Ceiling on helper *silence* during one call, so a wedged helper cannot stall a program. */
const CALL_TIMEOUT_MS = 120_000;

/**
 * Backoff before retrying a request the model manager turned away as busy
 * (several processes using the model at once). Observed to clear within a
 * second under a full parallel test run.
 */
const BUSY_RETRY_DELAYS_MS = [250, 1000, 2500];

/** What the model can actually do, as reported by the framework (macOS 27+). */
export interface ModelCapabilities {
  vision: boolean;
  guidedGeneration: boolean;
  reasoning: boolean;
  toolCalling: boolean;
}

/** Feature flags reported by --probe (absent on older helpers). */
export interface HelperFeatures {
  protocol?: number;
  streaming: boolean;
  structuredStreaming?: boolean;
  sessions: boolean;
  history: boolean;
  transcripts?: boolean;
  labelledAttachments: boolean;
  functionTools?: boolean;
  cancellation?: boolean;
  builtInTools: string[];
}

/** An image attachment: a bare path, or a path with a Siri-style label. */
export type ImageAttachment = string | { path: string; label?: string };

/** A history turn mirrored by the helper for a named session. */
export interface HistoryTurn {
  role: string;
  content: string;
}

/**
 * Private Cloud Compute quota, read from the framework without calling it.
 *
 * PCC *inference* needs an entitlement no installable package can ship, which is
 * why the cloud tier goes through Shortcuts — but `quotaUsage` is readable from
 * an unentitled process, so this is real first-party quota state rather than a
 * guess parsed out of an error message.
 */
export interface CloudQuota {
  isAvailable: boolean;
  status: 'belowLimit' | 'limitReached' | 'unknown';
  approachingLimit?: boolean;
  resetDate?: string;
}

/**
 * The Private Cloud Compute model as the framework describes it (macOS 27+):
 * quota, capabilities and context size, all readable without the entitlement
 * that blocks calling it directly.
 */
export interface CloudModelInfo extends CloudQuota {
  capabilities?: ModelCapabilities;
  contextSize?: number;
}

export interface DeviceProbe {
  available: boolean;
  reason?: string;
  contextSize?: number;
  variant?: string;
  /** macOS 27+ only; absent on macOS 26. */
  capabilities?: ModelCapabilities;
  useCases?: string[];
  /** Helper feature flags; absent when the cached binary predates them. */
  features?: HelperFeatures;
  /** PCC quota and model info, surfaced here because the on-device helper is what can read them. */
  cloud?: CloudModelInfo;
}

/**
 * How tokens are chosen.
 *
 * `greedy` is deterministic but degenerates under guided generation — it is the
 * `temperature: 0` trap by another name. The seeded modes give the same
 * reproducibility *without* that failure: `{ mode: 'topK', k: 50, seed: 42 }`
 * returned byte-identical output across three runs here. Determinism relies on
 * a fresh session per request, which is the default.
 */
export type SamplingMode =
  | { mode: 'greedy' }
  | { mode: 'topK'; k?: number; seed?: number }
  | { mode: 'threshold'; p?: number; seed?: number };

/** Apple ships a use case specialised for tagging and topic extraction. */
export type UseCase = 'general' | 'contentTagging';

/**
 * `permissive` selects `permissiveContentTransformations`, which relaxes the
 * guardrails for content *transformation* — rewriting or summarising text the
 * default guardrails would refuse to touch.
 */
export type Guardrails = 'default' | 'permissive';

/** Token accounting for one generation (macOS 27+). */
export interface Usage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  /** Prompt tokens served from the session's cache — a conversation's earlier turns. */
  cachedInputTokens?: number;
}

/** Why generation ended. `length` means `maxTokens` cut it off; `tool-calls` means a client-driven tool is waiting. */
export type FinishReason = 'stop' | 'length' | 'tool-calls';

/** One tool call the model made while producing a reply. */
export interface ToolCallRecord {
  id: string;
  name: string;
  arguments: unknown;
  /** What the tool returned to the model. Absent for a client-driven call not yet run. */
  output?: string;
}

interface HelperResponse {
  ok?: boolean;
  content?: string;
  delta?: string;
  done?: boolean;
  error?: string;
  kind?: string;
  resetDate?: string;
  tokens?: number;
  tokenCount?: number;
  contextSize?: number;
  prewarmed?: boolean;
  reset?: boolean;
  history?: HistoryTurn[];
  instructions?: string;
  sessionId?: string;
  finishReason?: string;
  usage?: { inputTokens?: number; outputTokens?: number; cachedInputTokens?: number };
  toolCalls?: Array<{ id?: string; name?: string; arguments?: unknown; output?: string }>;
  trimmedTurns?: number;
  tool?: string;
}

export interface DeviceRequest {
  system?: string;
  prompt: string;
  schema?: JsonSchema | null;
  temperature?: number;
  maxTokens?: number;
  /** Image attachments. Entries may carry a label for follow-up turns. */
  images?: ImageAttachment[];
  /** Text documents inlined into the prompt client-side (see withDocuments). */
  documents?: string[];
  /** Named multi-turn conversation; the helper keeps the native transcript. */
  sessionId?: string;
  /** Earlier turns, sent as a native transcript (stateless multi-turn). */
  history?: HistoryEntry[];
  /** Drop the oldest history turns rather than fail when they do not fit. */
  trimHistory?: boolean;
  /** Built-in Apple tools: on-device OCR, barcode reading, Spotlight RAG. */
  tools?: BuiltInTool[];
  /** Function tools, already normalised (see normalizeTools). */
  functions?: NormalizedTools['functions'];
  useCase?: UseCase;
  guardrails?: Guardrails;
  sampling?: SamplingMode;
  /** Send the schema in the prompt as well as to the decoder. See helper.swift. */
  includeSchemaInPrompt?: boolean;
  /**
   * Reuse one session across calls. Off by default: it makes unrelated calls
   * share a transcript. See the note in helper.swift.
   * Prefer sessionId for conversations; this flag is the legacy single slot.
   */
  reuseSession?: boolean;
}

/** Per-call hooks for `run`. */
export interface RunHooks {
  signal?: AbortSignal;
  onDelta?: (delta: string) => void;
  /** Each partial JSON document while streaming with a schema. */
  onPartial?: (json: string) => void;
  /** Every tool call, as the model makes it. */
  onToolCall?: (call: ToolCallRecord) => void;
  /** Results the caller already has for calls the model may repeat. */
  replay?: ReplayBook;
  maxToolCalls?: number;
}

/** What one generation produced. */
export interface DeviceOutcome {
  content: string;
  finishReason: FinishReason;
  usage?: Usage;
  toolCalls: ToolCallRecord[];
  trimmedTurns: number;
}

/** Assert every requested tool is one the helper knows, before spending a call. */
export function assertTools(tools: BuiltInTool[] | undefined): void {
  if (tools === undefined) return;
  for (const tool of tools) {
    if (!BUILT_IN_TOOLS.has(tool)) {
      throw new AppleLLMError(
        `Unknown tool "${tool}" (want ${[...BUILT_IN_TOOLS].join(', ')}).`,
        'device',
      );
    }
  }
}

/** Probe the on-device model without constructing a client. */
export async function probeDevice(onProgress?: OnProgress): Promise<DeviceProbe> {
  if (process.platform !== 'darwin') {
    return { available: false, reason: `unsupportedPlatform: this machine reports "${process.platform}"` };
  }
  if (!isAppleSiliconMac()) {
    return { available: false, reason: 'deviceNotEligible: Apple Intelligence needs Apple Silicon' };
  }
  const major = await macosMajor();
  // Checked before compiling: on macOS 25 the compile fails with an opaque SDK
  // error, and `probe()` must return a reason rather than burn a build attempt.
  if (major !== null && major < 26) {
    return { available: false, reason: `unsupportedOSVersion: needs macOS 26 or later, this is ${major}` };
  }

  let binary: string;
  try {
    binary = await ensureBinary(onProgress);
  } catch (err) {
    return { available: false, reason: err instanceof Error ? err.message : String(err) };
  }

  try {
    const { stdout } = await execFileAsync(binary, ['--probe'], { timeout: 20_000 });
    return JSON.parse(stdout.trim()) as DeviceProbe;
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    return { available: false, reason: `the helper could not be run: ${reason.slice(0, 300)}` };
  }
}

/** Turn a helper failure envelope into the matching typed error. */
function raiseFor(response: HelperResponse, raw: string, toolFailure?: { name: string; error: unknown }): never {
  const detail = response.error ?? raw.slice(0, 200);
  switch (response.kind) {
    case 'availability':
      throw new ModelUnavailableError(
        unavailableMessage(detail),
        toUnavailableReason(detail),
        'device',
      );
    case 'schema':
      throw new SchemaRejectedError(
        `Apple rejected the response schema: ${detail}\n` +
          'Every object needs a title, an x-order and additionalProperties; enums must be anyOf/const; unions cannot include null.',
        'device',
      );
    case 'context':
      throw new ContextLengthError(
        `The prompt exceeded the on-device context window: ${detail}`,
        'device',
        response.contextSize,
        response.tokenCount,
      );
    case 'quota': {
      // Python validates the timestamp; an unparseable resetDate must not
      // become an Invalid Date.
      const parsed = response.resetDate === undefined ? undefined : new Date(response.resetDate);
      const resetDate = parsed !== undefined && !Number.isNaN(parsed.getTime()) ? parsed : undefined;
      throw new QuotaError(
        `Apple rate limited the on-device model: ${detail}`,
        'device',
        resetDate,
      );
    }
    case 'guardrail':
      throw new RefusalError(`The on-device model declined to answer: ${detail}`, 'device');
    case 'timeout':
      throw new TimeoutError(`Apple on-device generation timed out: ${detail}`, 'device');
    case 'cancelled':
      throw new AbortError('The request was cancelled.', 'device');
    case 'tool': {
      const name = toolFailure?.name ?? response.tool ?? 'unknown';
      const cause = toolFailure?.error;
      const why = cause instanceof Error ? cause.message : cause === undefined ? detail : String(cause);
      throw new ToolExecutionError(`Tool "${name}" failed: ${why}`, name, { cause });
    }
    case 'unsupported':
      throw new UnsupportedError(`Apple on-device generation failed: ${detail}`, 'device');
    case 'busy':
      throw new ModelBusyError(
        'Apple’s on-device model is busy serving other processes and turned the request away ' +
          '(ModelManagerError 1042), even after retrying. Try again shortly.',
        'device',
      );
    default:
      throw new AppleLLMError(`Apple on-device generation failed: ${detail}`, 'device');
  }
}

function usageFrom(raw: HelperResponse['usage']): Usage | undefined {
  if (raw === undefined || typeof raw.inputTokens !== 'number' || typeof raw.outputTokens !== 'number') {
    return undefined;
  }
  return {
    inputTokens: raw.inputTokens,
    outputTokens: raw.outputTokens,
    totalTokens: raw.inputTokens + raw.outputTokens,
    ...(typeof raw.cachedInputTokens === 'number' ? { cachedInputTokens: raw.cachedInputTokens } : {}),
  };
}

function finishFrom(raw: string | undefined): FinishReason {
  if (raw === 'length') return 'length';
  if (raw === 'toolCalls') return 'tool-calls';
  return 'stop';
}

/**
 * The on-device tier: Apple's `FoundationModels` framework, reached through a
 * self-compiled Swift helper. Nothing leaves the machine.
 */
export interface DeviceClientOptions {
  useCase?: UseCase;
  guardrails?: Guardrails;
  /**
   * A ready helper to use instead of probing and compiling one. For tests,
   * which script a fake helper; not needed in normal use.
   */
  server?: HelperServer;
}

export class DeviceClient {
  private binary?: string;
  private probeResult?: DeviceProbe;
  private server?: HelperServer;
  private readying?: Promise<void>;

  constructor(private readonly options: DeviceClientOptions = {}) {
    this.server = options.server;
  }

  get label(): string {
    const variant = this.probeResult?.variant;
    return variant ? `apple on-device (${variant})` : 'apple on-device';
  }

  get contextSize(): number | undefined {
    return this.probeResult?.contextSize;
  }

  /** Probe and build once; concurrent first calls share the one attempt. */
  async ensureReady(onProgress?: OnProgress): Promise<void> {
    if (this.probeResult?.available === true && this.binary !== undefined) return;
    this.readying ??= (async () => {
      const probe = await probeDevice(onProgress);
      this.probeResult = probe;
      if (!probe.available) {
        throw new ModelUnavailableError(
          unavailableMessage(probe.reason),
          toUnavailableReason(probe.reason),
          'device',
        );
      }
      this.binary = await ensureBinary(onProgress);
      onProgress?.({
        status: `apple on-device model ready${probe.variant ? ` (${probe.variant}, ${probe.contextSize} token context)` : ''}`,
      });
    })().finally(() => {
      this.readying = undefined;
    });
    return this.readying;
  }

  getProbe(): DeviceProbe | undefined {
    return this.probeResult;
  }

  private async helper(): Promise<HelperServer> {
    if (this.options.server !== undefined) return this.options.server;
    await this.ensureReady();
    const binary = this.binary ?? (await ensureBinary());
    this.server ??= new HelperServer(binary, { timeoutMs: CALL_TIMEOUT_MS });
    return this.server;
  }

  /** Send one single-line envelope and return the parsed reply, raising a typed error on failure. */
  private async exchange(envelope: Record<string, unknown>, signal?: AbortSignal): Promise<HelperResponse> {
    const server = await this.helper();
    const id = randomUUID();
    const raw = await server.request(JSON.stringify({ ...envelope, id }), { signal, id });
    const response = parse(raw);
    if (response.ok !== true) raiseFor(response, raw);
    return response;
  }

  /** The request envelope, shared by every generation path. */
  private async envelopeFor(request: DeviceRequest, op: 'generate' | 'stream'): Promise<Record<string, unknown>> {
    assertTools(request.tools);
    return {
      op,
      instructions: request.system ?? '',
      prompt: await withDocuments(request.prompt, request.documents),
      schema: request.schema ?? null,
      temperature: request.temperature ?? DEFAULT_TEMPERATURE,
      maxTokens: request.maxTokens ?? DEFAULT_MAX_TOKENS,
      includeSchemaInPrompt: request.includeSchemaInPrompt,
      reuseSession: request.reuseSession ?? false,
      sessionId: request.sessionId,
      history: request.history !== undefined && request.history.length > 0 ? request.history : undefined,
      trimHistory: request.trimHistory,
      tools: request.tools !== undefined && request.tools.length > 0 ? request.tools : undefined,
      functions:
        request.functions !== undefined && request.functions.length > 0
          ? request.functions.map((fn) => ({
              name: fn.name,
              description: fn.description ?? '',
              parameters: toAppleSchema(fn.json, `${fn.name}Arguments`),
            }))
          : undefined,
      images: request.images,
      useCase: request.useCase ?? this.options.useCase,
      guardrails: request.guardrails ?? this.options.guardrails,
      sampling: request.sampling,
    };
  }

  /**
   * One generation, with every event handled: text deltas, partial objects,
   * and function tool calls, which run here and answer the helper.
   */
  async run(request: DeviceRequest, op: 'generate' | 'stream' = 'generate', hooks: RunHooks = {}): Promise<DeviceOutcome> {
    const envelope = await this.envelopeFor(request, op);
    const server = await this.helper();
    const functions = new Map((request.functions ?? []).map((fn) => [fn.name, fn]));
    const maxToolCalls = hooks.maxToolCalls ?? DEFAULT_MAX_TOOL_CALLS;
    const calls: ToolCallRecord[] = [];
    let toolFailure: { name: string; error: unknown } | undefined;

    const answerToolCall = async (
      call: ToolCallRecord,
      write: (control: Record<string, unknown>) => void,
    ): Promise<void> => {
      const reply = (fields: Record<string, unknown>): void => write({ op: 'toolResult', callId: call.id, ...fields });
      const fn = functions.get(call.name);
      if (fn === undefined) {
        reply({ output: `There is no tool named "${call.name}".`, isError: true });
        return;
      }
      if (calls.length > maxToolCalls) {
        call.output = 'Tool call limit reached. Answer with the information you already have.';
        reply({ output: call.output });
        return;
      }
      const replayed = hooks.replay?.take(call.name, call.arguments);
      if (replayed !== undefined) {
        call.output = replayed;
        reply({ output: replayed });
        return;
      }
      if (fn.execute === undefined) {
        // Client-driven: the caller runs it and continues with a tool message.
        reply({ stop: true });
        return;
      }
      let args: unknown = restoreNulls(call.arguments, fn.json);
      if (fn.resolved?.validate !== undefined) {
        const checked = await fn.resolved.validate(args);
        if (checked.issues !== undefined) {
          // Constrained decoding already matched the shape; this is a
          // refinement the decoder cannot express. Let the model see it and retry.
          call.output = `Invalid arguments: ${describeIssues(checked.issues)}`;
          reply({ output: call.output });
          return;
        }
        args = checked.value;
      }
      server.pause();
      try {
        const result = await fn.execute(args, { toolCallId: call.id, signal: hooks.signal });
        call.output = toolOutputText(result);
        reply({ output: call.output });
      } catch (error) {
        toolFailure = { name: call.name, error };
        reply({ output: error instanceof Error ? error.message : String(error), isError: true });
      } finally {
        server.resume();
      }
    };

    let delivered = false;
    // One id per attempt, in the envelope and the hooks alike: it is what a
    // cancel line targets.
    const attempt = (id = randomUUID()): Promise<string> =>
      server.request(JSON.stringify({ ...envelope, id }), {
        id,
        signal: hooks.signal,
        onEvent: (event, write) => {
          delivered = true;
          if (typeof event.delta === 'string') {
            if (event.delta !== '') hooks.onDelta?.(event.delta);
          } else if (typeof event.partial === 'string') {
            hooks.onPartial?.(event.partial);
          } else if (event.toolCall !== null && typeof event.toolCall === 'object') {
            const toolCall = event.toolCall as { id?: unknown; name?: unknown; arguments?: unknown };
            const call: ToolCallRecord = {
              id: String(toolCall.id ?? randomUUID()),
              name: String(toolCall.name ?? ''),
              arguments: toolCall.arguments ?? {},
            };
            calls.push(call);
            try {
              hooks.onToolCall?.(call);
            } catch {
              // An observer must not break the tool loop.
            }
            void answerToolCall(call, write).catch((error: unknown) => {
              toolFailure ??= { name: call.name, error };
              write({ op: 'toolResult', callId: call.id, output: String(error), isError: true });
            });
          }
        },
      });

    let raw = await attempt();
    let response = parse(raw);
    // A busy model manager rejects before generating anything, so retrying is
    // safe — unless this request already produced events a caller has seen.
    for (const delay of BUSY_RETRY_DELAYS_MS) {
      if (response.ok === true || response.kind !== 'busy' || delivered) break;
      await sleep(delay, hooks.signal);
      raw = await attempt();
      response = parse(raw);
    }
    if (response.ok !== true) raiseFor(response, raw, toolFailure);
    if (typeof response.content !== 'string') {
      throw new AppleLLMError('Apple helper returned no content.', 'device');
    }
    // Function tool calls as this process saw and answered them (their ids
    // are the ones `execute` was given); built-in tool calls as the helper's
    // transcript recorded them.
    const builtIn = (response.toolCalls ?? [])
      .filter((c) => typeof c.name === 'string' && !functions.has(c.name))
      .map((c) => ({
        id: String(c.id ?? ''),
        name: String(c.name),
        arguments: c.arguments ?? {},
        ...(typeof c.output === 'string' ? { output: c.output } : {}),
      }));
    return {
      content: response.content,
      finishReason: finishFrom(response.finishReason),
      usage: usageFrom(response.usage),
      toolCalls: [...calls, ...builtIn],
      trimmedTurns: typeof response.trimmedTurns === 'number' ? response.trimmedTurns : 0,
    };
  }

  /** One request. Returns the helper's `content` string, unparsed. */
  async complete(request: DeviceRequest, hooks: RunHooks = {}): Promise<string> {
    return (await this.run(request, 'generate', hooks)).content;
  }

  /**
   * Streaming text. Deltas arrive via onDelta as the model generates; the
   * promise resolves with the full text.
   */
  async stream(
    prompt: string,
    options: Omit<DeviceRequest, 'prompt' | 'schema'> & { onDelta?: (delta: string) => void; signal?: AbortSignal } = {},
  ): Promise<string> {
    const { onDelta, signal, ...rest } = options;
    return (await this.run({ ...rest, prompt, schema: null }, 'stream', { onDelta, signal })).content;
  }

  /**
   * Conversation history for a named session: the mirrored turns the helper
   * persisted, oldest first. Survives helper restarts, and the next call
   * rebuilds the native transcript from it.
   */
  async history(sessionId: string): Promise<{ instructions: string; history: HistoryTurn[] }> {
    const response = await this.exchange({ op: 'history', sessionId, instructions: '' });
    return {
      instructions: typeof response.instructions === 'string' ? response.instructions : '',
      history: Array.isArray(response.history) ? response.history : [],
    };
  }

  /** Drop a named session (and its persisted history), or all of them. */
  async resetSession(sessionId?: string): Promise<void> {
    await this.exchange({ op: 'reset', sessionId: sessionId ?? '', instructions: '' });
  }

  /**
   * How many tokens a request costs, before sending it.
   *
   * The point is to turn a ContextLengthError into an arithmetic check: compare
   * against `contextSize` and trim, rather than discovering the ceiling by
   * hitting it. Counts everything that shares the window: instructions,
   * history, tools, schema, images and inlined documents.
   */
  async countTokens(
    prompt: string,
    options: {
      system?: string;
      images?: ImageAttachment[];
      tools?: BuiltInTool[];
      documents?: string[];
      history?: HistoryEntry[];
      schema?: JsonSchema;
      functions?: NormalizedTools['functions'];
      signal?: AbortSignal;
    } = {},
  ): Promise<{ tokens: number; contextSize: number }> {
    const envelope = await this.envelopeFor(
      {
        prompt,
        system: options.system,
        images: options.images,
        tools: options.tools,
        documents: options.documents,
        history: options.history,
        schema: options.schema === undefined ? null : toAppleSchema(options.schema),
        functions: options.functions,
      },
      'generate',
    );
    const response = await this.exchange({ ...envelope, op: 'countTokens' }, options.signal);
    if (typeof response.tokens !== 'number' || typeof response.contextSize !== 'number') {
      throw new AppleLLMError('Apple helper returned an unreadable token count.', 'device');
    }
    return { tokens: response.tokens, contextSize: response.contextSize };
  }

  /**
   * Load the model assets now so the first real call does not pay for it.
   *
   * Cheap and idempotent, but do not expect much on a warm machine: with the
   * assets already resident this measured 0.31s against 0.36s for an
   * unprewarmed first call — inside the noise. The win is on a genuinely cold
   * system, where the very first call to the framework here took 7.8s. Worth
   * calling at startup when you know a request is coming; not worth building
   * around.
   */
  async prewarm(system?: string): Promise<void> {
    await this.exchange({ op: 'prewarm', instructions: system ?? '' });
  }

  async text(prompt: string, options: Omit<DeviceRequest, 'prompt' | 'schema'> = {}): Promise<string> {
    return this.complete({ ...options, prompt, schema: null });
  }

  async json(prompt: string, options: Omit<DeviceRequest, 'prompt'> & { schema: JsonSchema }): Promise<unknown> {
    const content = await this.complete({ ...options, prompt, schema: toAppleSchema(options.schema) });
    try {
      return restoreNulls(JSON.parse(content), options.schema);
    } catch {
      // Constrained decoding should make this unreachable.
      throw new AppleLLMError(
        `Apple on-device model returned invalid JSON: ${content.slice(0, 200)}`,
        'device',
      );
    }
  }

  /** Shut the helper process down. Safe to call more than once. */
  close(): void {
    this.server?.stop();
    this.server = undefined;
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(abortErrorFrom(signal));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(abortErrorFrom(signal!));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

function parse(raw: string): HelperResponse {
  try {
    return JSON.parse(raw) as HelperResponse;
  } catch {
    throw new AppleLLMError(`Apple helper returned an unreadable envelope: ${raw.slice(0, 200)}`, 'device');
  }
}

/**
 * Inline text documents into the prompt client-side.
 *
 * The helper's vision path handles images; plain-text sources (.txt/.md/.json
 * and friends) are cheaper to splice here than to teach the Swift side about.
 * Binary files are refused loudly — silently skipping a document the caller
 * asked about would be the vision-drop trap by another name.
 */
export async function withDocuments(prompt: string, documents: string[] | undefined): Promise<string> {
  if (documents === undefined || documents.length === 0) return prompt;
  const { readFile, stat } = await import('node:fs/promises');
  const parts: string[] = [prompt];
  for (const doc of documents) {
    let size = 0;
    try {
      size = (await stat(doc)).size;
    } catch {
      throw new AppleLLMError(`Document not found: ${doc}`);
    }
    if (size > 512_000) {
      throw new AppleLLMError(`Document too large to inline (>${512_000} bytes): ${doc}`);
    }
    let text: string;
    try {
      text = await readFile(doc, 'utf8');
    } catch {
      throw new AppleLLMError(`Document is not readable text: ${doc}`);
    }
    if (text.includes('�')) {
      throw new AppleLLMError(`Document is not readable text: ${doc}`);
    }
    parts.push(`\n\n--- Document: ${doc} ---\n${text}`);
  }
  return parts.join('');
}

/** Parse a CLI --image value: "path" or "path::label". */
export function parseImageFlag(value: string): ImageAttachment {
  const sep = value.lastIndexOf('::');
  if (sep > 0) {
    const path = value.slice(0, sep);
    const label = value.slice(sep + 2).trim();
    if (path !== '' && label !== '') return { path, label };
  }
  return value;
}

export { ReplayBook };
