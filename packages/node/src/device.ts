import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ensureBinary, type OnProgress } from './compile.js';
import {
  ContextLengthError,
  ModelUnavailableError,
  QuotaError,
  RefusalError,
  SchemaRejectedError,
  AppleLLMError,
  toUnavailableReason,
  unavailableMessage,
} from './errors.js';
import { HelperServer } from './protocol.js';
import { toAppleSchema, type JsonSchema } from './schema.js';
import { isAppleSiliconMac, macosMajor } from './target.js';

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

/** Hard ceiling on one call, so a wedged helper cannot stall a program. */
const CALL_TIMEOUT_MS = 120_000;

/** What the model can actually do, as reported by the framework (macOS 27+). */
export interface ModelCapabilities {
  vision: boolean;
  guidedGeneration: boolean;
  reasoning: boolean;
  toolCalling: boolean;
}

/** Siri-parity feature flags reported by --probe (absent on older helpers). */
export interface HelperFeatures {
  streaming: boolean;
  sessions: boolean;
  history: boolean;
  labelledAttachments: boolean;
  builtInTools: string[];
}

/** An image attachment: a bare path, or a path with a Siri-style label. */
export type ImageAttachment = string | { path: string; label?: string };

/** Apple's built-in on-device tools (all macOS 27+, all local). */
export type BuiltInTool = 'ocr' | 'barcode' | 'spotlight';

const BUILT_IN_TOOLS: ReadonlySet<string> = new Set(['ocr', 'barcode', 'spotlight']);

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
  /** PCC quota, surfaced here because the on-device helper is what can read it. */
  cloud?: CloudQuota;
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

interface HelperResponse {
  ok?: boolean;
  content?: string;
  delta?: string;
  done?: boolean;
  error?: string;
  kind?: string;
  resetDate?: string;
  tokens?: number;
  contextSize?: number;
  prewarmed?: boolean;
  reset?: boolean;
  history?: HistoryTurn[];
  instructions?: string;
  sessionId?: string;
}

export interface DeviceRequest {
  system?: string;
  prompt: string;
  schema?: JsonSchema | null;
  temperature?: number;
  maxTokens?: number;
  /** Image attachments. Entries may carry a label for follow-up turns. */
  images?: ImageAttachment[];
  /** Text documents inlined into the prompt client-side (see inlineDocuments). */
  documents?: string[];
  /** Named multi-turn conversation; the helper keeps the native transcript. */
  sessionId?: string;
  /** Built-in Apple tools: on-device OCR, barcode reading, Spotlight RAG. */
  tools?: BuiltInTool[];
  useCase?: UseCase;
  guardrails?: Guardrails;
  sampling?: SamplingMode;
  /** Send the schema in the prompt as well as to the decoder. See helper.swift. */
  includeSchemaInPrompt?: boolean;
  /**
   * Reuse one session across calls, as api-scribe did. Off by default: it makes
   * unrelated calls share a transcript. See the note in helper.swift.
   * Prefer sessionId for conversations; this flag is the legacy single slot.
   */
  reuseSession?: boolean;
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
function raiseFor(response: HelperResponse, raw: string): never {
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
      throw new ContextLengthError(`The prompt exceeded the on-device context window: ${detail}`, 'device');
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
    default:
      throw new AppleLLMError(`Apple on-device generation failed: ${detail}`, 'device');
  }
}

/**
 * The on-device tier: Apple's `FoundationModels` framework, reached through a
 * self-compiled Swift helper. Nothing leaves the machine.
 */
export interface DeviceClientOptions {
  useCase?: UseCase;
  guardrails?: Guardrails;
}

export class DeviceClient {
  private binary?: string;
  private probeResult?: DeviceProbe;
  private server?: HelperServer;

  constructor(private readonly options: DeviceClientOptions = {}) {}

  get label(): string {
    const variant = this.probeResult?.variant;
    return variant ? `apple on-device (${variant})` : 'apple on-device';
  }

  get contextSize(): number | undefined {
    return this.probeResult?.contextSize;
  }

  async ensureReady(onProgress?: OnProgress): Promise<void> {
    if (this.probeResult?.available === true) return;
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
  }

  getProbe(): DeviceProbe | undefined {
    return this.probeResult;
  }

  /** Send one envelope and return the parsed reply, raising a typed error on failure. */
  private async exchange(envelope: Record<string, unknown>): Promise<HelperResponse> {
    await this.ensureReady();
    const binary = this.binary ?? (await ensureBinary());
    this.server ??= new HelperServer(binary, { timeoutMs: CALL_TIMEOUT_MS });
    const raw = await this.server.send(JSON.stringify(envelope));

    let response: HelperResponse;
    try {
      response = JSON.parse(raw) as HelperResponse;
    } catch {
      throw new AppleLLMError(`Apple helper returned an unreadable envelope: ${raw.slice(0, 200)}`, 'device');
    }
    if (response.ok !== true) raiseFor(response, raw);
    return response;
  }

  /** One request. Returns the helper's `content` string, unparsed. */
  async complete(request: DeviceRequest): Promise<string> {
    assertTools(request.tools);
    const prompt = await withDocuments(request.prompt, request.documents);
    const response = await this.exchange({
      instructions: request.system ?? '',
      prompt,
      schema: request.schema ?? null,
      temperature: request.temperature ?? DEFAULT_TEMPERATURE,
      maxTokens: request.maxTokens ?? DEFAULT_MAX_TOKENS,
      includeSchemaInPrompt: request.includeSchemaInPrompt,
      reuseSession: request.reuseSession ?? false,
      sessionId: request.sessionId,
      tools: request.tools,
      images: request.images,
      useCase: request.useCase ?? this.options.useCase,
      guardrails: request.guardrails ?? this.options.guardrails,
      sampling: request.sampling,
    });
    if (typeof response.content !== 'string') {
      throw new AppleLLMError('Apple helper returned no content.', 'device');
    }
    return response.content;
  }

  /**
   * Streaming text. Deltas arrive via onDelta as the model generates; the
   * promise resolves with the full text. Text only: the helper rejects
   * schema+stream, because partial JSON is not a usable delta.
   */
  async stream(
    prompt: string,
    options: Omit<DeviceRequest, 'prompt' | 'schema'> & { onDelta?: (delta: string) => void } = {},
  ): Promise<string> {
    assertTools(options.tools);
    const { onDelta, documents, ...rest } = options;
    const fullPrompt = await withDocuments(prompt, documents);
    await this.ensureReady();
    const binary = this.binary ?? (await ensureBinary());
    this.server ??= new HelperServer(binary, { timeoutMs: CALL_TIMEOUT_MS });
    const envelope = {
      op: 'stream',
      instructions: rest.system ?? '',
      prompt: fullPrompt,
      schema: null,
      temperature: rest.temperature ?? DEFAULT_TEMPERATURE,
      maxTokens: rest.maxTokens ?? DEFAULT_MAX_TOKENS,
      reuseSession: rest.reuseSession ?? false,
      sessionId: rest.sessionId,
      tools: rest.tools,
      images: rest.images,
      useCase: rest.useCase ?? this.options.useCase,
      guardrails: rest.guardrails ?? this.options.guardrails,
      sampling: rest.sampling,
    };
    const raw = await this.server.stream(JSON.stringify(envelope), (delta) => {
      if (delta !== '') onDelta?.(delta);
    });
    let response: HelperResponse;
    try {
      response = JSON.parse(raw) as HelperResponse;
    } catch {
      throw new AppleLLMError(`Apple helper returned an unreadable envelope: ${raw.slice(0, 200)}`, 'device');
    }
    if (response.ok !== true) raiseFor(response, raw);
    if (typeof response.content !== 'string') {
      throw new AppleLLMError('Apple helper returned no content.', 'device');
    }
    return response.content;
  }

  /**
   * Conversation history for a named session: the mirrored turns the helper
   * persisted, oldest first. Survives helper restarts; the native transcript
   * does not, so treat a restart as a context break, not a loss.
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
   * How many tokens a prompt costs, before sending it.
   *
   * The point is to turn a ContextLengthError into an arithmetic check: compare
   * against `contextSize` and trim, rather than discovering the ceiling by
   * hitting it. Counts the instructions too, since they share the window.
   */
  async countTokens(
    prompt: string,
    options: { system?: string; images?: ImageAttachment[]; tools?: BuiltInTool[] } = {},
  ): Promise<{ tokens: number; contextSize: number }> {
    const response = await this.exchange({
      op: 'countTokens',
      instructions: options.system ?? '',
      prompt: await withDocuments(prompt, undefined),
      images: options.images,
      tools: options.tools,
    });
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
      return JSON.parse(content);
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
      throw new AppleLLMError(`Document not found: ${doc}`, 'device');
    }
    if (size > 512_000) {
      throw new AppleLLMError(`Document too large to inline (>${512_000} bytes): ${doc}`, 'device');
    }
    let text: string;
    try {
      text = await readFile(doc, 'utf8');
    } catch {
      throw new AppleLLMError(`Document is not readable text: ${doc}`, 'device');
    }
    if (text.includes('�')) {
      throw new AppleLLMError(`Document is not readable text: ${doc}`, 'device');
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
