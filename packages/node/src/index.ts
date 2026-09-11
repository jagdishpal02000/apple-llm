/**
 * apple-llm — one library for Apple's on-device and Private Cloud Compute models.
 *
 * macOS 26+ on Apple Silicon only. No API key, no account, no developer program.
 * Extracted from api-scribe (MIT), which discovered and shipped both routes.
 */
import { probeCloud, CloudClient, CLOUD_CONTEXT_TOKENS, cloudSetupHint, type CloudProbe } from './cloud.js';
import {
  DeviceClient,
  assertTools,
  parseImageFlag,
  probeDevice,
  withDocuments,
  type BuiltInTool,
  type DeviceProbe,
  type DeviceRequest,
  type Guardrails,
  type HistoryTurn,
  type ImageAttachment,
  type SamplingMode,
  type UseCase,
} from './device.js';
import type { OnProgress } from './compile.js';
import { AppleLLMError, ModelUnavailableError, SetupRequiredError } from './errors.js';
import type { JsonSchema } from './schema.js';

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
  // The on-device helper is the only thing that can read the PCC quota — the
  // entitlement that blocks PCC *inference* does not block reading `quotaUsage`
  // — so the cloud picture is assembled from both probes.
  if (device.cloud !== undefined) cloud.quota = device.cloud;
  return { device, cloud };
}

export interface AppleLLMOptions {
  tier?: Tier;
  /** Default sampling temperature. See DEFAULT_TEMPERATURE — do not set 0. */
  temperature?: number;
  maxTokens?: number;
  /** Called for the one-time compile and the shortcut install; both take seconds. */
  onProgress?: OnProgress;
  /** `contentTagging` selects Apple's tagging-specialised model. Device tier only. */
  useCase?: UseCase;
  /** `permissive` relaxes guardrails for rewriting tasks. Device tier only. */
  guardrails?: Guardrails;
  /** Default sampling mode; a seeded one makes output reproducible. */
  sampling?: SamplingMode;
}

export interface TextOptions {
  system?: string;
  temperature?: number;
  maxTokens?: number;
  /** Cloud tier only: run with "Use Broad World Knowledge". */
  webSearch?: boolean;
  /** Image attachments; entries may carry a label for follow-up turns. */
  images?: ImageAttachment[];
  /** Text documents inlined into the prompt (device tier). */
  documents?: string[];
  /** Named multi-turn conversation (device tier). */
  sessionId?: string;
  /** Built-in Apple tools: ocr, barcode, spotlight (device, macOS 27+). */
  tools?: BuiltInTool[];
  useCase?: UseCase;
  guardrails?: Guardrails;
  sampling?: SamplingMode;
}

export interface StreamOptions extends TextOptions {
  onDelta?: (delta: string) => void;
}

export interface JsonOptions extends TextOptions {
  schema: JsonSchema;
}

/**
 * The main entry point.
 *
 *   const llm = new AppleLLM({ tier: 'device' });
 *   await llm.text('Summarize this');
 *   await llm.json('Extract the fields', { schema });
 *   llm.close();
 */
export class AppleLLM {
  private device?: DeviceClient;
  private cloud?: CloudClient;
  /** Which tier `auto` settled on, once resolved. */
  private resolved?: 'device' | 'cloud';
  private readonly options: AppleLLMOptions;

  constructor(options: AppleLLMOptions = {}) {
    this.options = options;
  }

  /** Per-call device settings, falling back to the constructor defaults. */
  private deviceDefaults(
    options: TextOptions,
  ): Omit<Partial<DeviceRequest>, 'prompt' | 'schema' | 'system'> {
    return {
      temperature: options.temperature ?? this.options.temperature,
      maxTokens: options.maxTokens ?? this.options.maxTokens,
      images: options.images,
      documents: options.documents,
      sessionId: options.sessionId,
      tools: options.tools,
      useCase: options.useCase ?? this.options.useCase,
      guardrails: options.guardrails ?? this.options.guardrails,
      sampling: options.sampling ?? this.options.sampling,
    };
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

  /**
   * Resolve the tier and do any one-time setup. Called automatically, but
   * exposed so a caller can pay the compile cost up front with a progress bar.
   */
  async ensureReady(onProgress?: OnProgress): Promise<void> {
    const progress = onProgress ?? this.options.onProgress;
    if (this.resolved !== undefined) return;

    if (this.tier === 'device') {
      this.device ??= new DeviceClient({
        useCase: this.options.useCase,
        guardrails: this.options.guardrails,
      });
      await this.device.ensureReady(progress);
      this.resolved = 'device';
      return;
    }
    if (this.tier === 'cloud') {
      this.cloud ??= new CloudClient();
      await this.cloud.ensureReady(progress);
      // Best effort: the device helper can read the PCC quota, which turns an
      // exhausted quota into an immediate error instead of a wasted round trip.
      this.cloud.setQuota((await probeDevice().catch(() => undefined))?.cloud);
      this.resolved = 'cloud';
      return;
    }

    // auto: on-device when available, else cloud when the shortcut is installed,
    // else an error naming the setup step.
    this.device ??= new DeviceClient({
      useCase: this.options.useCase,
      guardrails: this.options.guardrails,
    });
    try {
      await this.device.ensureReady(progress);
      this.resolved = 'device';
      return;
    } catch (deviceError) {
      this.cloud ??= new CloudClient();
      try {
        await this.cloud.ensureReady(progress);
        // Best effort: the device helper can read the PCC quota, which turns an
        // exhausted quota into an immediate error instead of a wasted round trip.
        this.cloud.setQuota((await probeDevice().catch(() => undefined))?.cloud);
        this.resolved = 'cloud';
        return;
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

  async text(prompt: string, options: TextOptions = {}): Promise<string> {
    await this.ensureReady();
    if (this.resolved === 'cloud') {
      if (options.images !== undefined && options.images.length > 0) {
        throw new AppleLLMError('images needs the on-device tier', 'cloud');
      }
      if (options.sessionId !== undefined) {
        throw new AppleLLMError('sessionId needs the on-device tier', 'cloud');
      }
      if (options.tools !== undefined && options.tools.length > 0) {
        throw new AppleLLMError('tools needs the on-device tier', 'cloud');
      }
      if (options.documents !== undefined && options.documents.length > 0) {
        throw new AppleLLMError('documents needs the on-device tier', 'cloud');
      }
      return this.cloud!.text({ system: options.system, prompt, webSearch: options.webSearch });
    }
    return this.device!.text(prompt, { system: options.system, ...this.deviceDefaults(options) });
  }

  /**
   * Streaming text (device tier only). Deltas arrive via onDelta as the model
   * generates; the promise resolves with the full text. The Siri-app shape:
   * partials first, final answer at the end.
   */
  async stream(prompt: string, options: StreamOptions = {}): Promise<string> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new AppleLLMError('stream needs the on-device tier.', 'cloud');
    }
    const { onDelta, ...rest } = options;
    return this.device!.stream(prompt, {
      system: rest.system,
      onDelta,
      ...this.deviceDefaults(rest),
    });
  }

  /** Conversation history for a named session (device tier only). */
  async history(sessionId: string): Promise<{ instructions: string; history: HistoryTurn[] }> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new AppleLLMError('history needs the on-device tier.', 'cloud');
    }
    return this.device!.history(sessionId);
  }

  /** Drop a named session, or all sessions when omitted (device tier only). */
  async resetSession(sessionId?: string): Promise<void> {
    await this.ensureReady();
    if (this.resolved === 'device') await this.device!.resetSession(sessionId);
  }

  /**
   * A named conversation: text/stream calls sharing one native transcript,
   * like one thread in the Siri app. History persists across helper restarts;
   * the native transcript does not (treated as a context break, not a loss).
   */
  conversation(sessionId: string, options: { system?: string } = {}): Conversation {
    return new Conversation(this, sessionId, options);
  }

  /**
   * Write with Siri, anywhere you type: drafting, rewriting and feedback
   * built on the permissive-content-transformation guardrails. Device tier
   * only — these are transformation tasks the default guardrails refuse.
   */
  async rewrite(text: string, options: { instruction?: string; system?: string } & TextOptions = {}): Promise<string> {
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

  async summarize(text: string, options: { length?: string } & TextOptions = {}): Promise<string> {
    const { length, ...rest } = options;
    return this.text(`Summarize the following text in ${length ?? 'one short paragraph'}.\n\n---\n${text}`, {
      ...rest,
      guardrails: rest.guardrails ?? 'permissive',
      system: rest.system ?? 'You summarize text tersely.',
    });
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
      throw new AppleLLMError('askScreen needs the on-device tier.', 'cloud');
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
   * Ask for JSON. On device the schema is *guaranteed* by constrained decoding.
   * On cloud it is requested in the prompt and recovered from the reply — the
   * cloud tier has no constrained decoding, so a bad shape is possible there.
   */
  async json(prompt: string, options: JsonOptions): Promise<unknown> {
    await this.ensureReady();
    if (this.resolved === 'cloud') {
      if (options.images !== undefined && options.images.length > 0) {
        throw new AppleLLMError('images needs the on-device tier', 'cloud');
      }
      return this.cloud!.json({
        system: options.system,
        prompt,
        schema: options.schema,
        webSearch: options.webSearch,
      });
    }
    return this.device!.json(prompt, {
      system: options.system,
      schema: options.schema,
      ...this.deviceDefaults(options),
    });
  }

  /**
   * api-scribe's `LlmClient` shape, so it can drop its four files and depend on
   * this instead. Not used internally.
   */
  async completeJson(system: string, user: string, schema: JsonSchema): Promise<unknown> {
    return this.json(user, { system, schema });
  }

  /**
   * How many tokens a prompt costs, before sending it. Device tier only.
   *
   * Turns a ContextLengthError into arithmetic: compare against `contextSize`
   * and trim, rather than finding the ceiling by hitting it.
   */
  async countTokens(
    prompt: string,
    options: { system?: string; images?: ImageAttachment[]; tools?: BuiltInTool[] } = {},
  ): Promise<{ tokens: number; contextSize: number }> {
    await this.ensureReady();
    if (this.resolved !== 'device') {
      throw new AppleLLMError('countTokens needs the on-device tier.', 'cloud');
    }
    return this.device!.countTokens(prompt, options);
  }

  /**
   * Load the model assets now so the first real call does not pay for it.
   * Device tier only; a no-op elsewhere. See `DeviceClient.prewarm` for what it
   * is actually worth (little, on a warm machine).
   */
  async prewarm(system?: string): Promise<void> {
    await this.ensureReady();
    if (this.resolved === 'device') await this.device!.prewarm(system);
  }

  /** Release the long-lived helper process. Safe to call more than once. */
  close(): void {
    this.device?.close();
    this.cloud?.close();
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

  async text(prompt: string, options: TextOptions = {}): Promise<string> {
    return this.llm.text(prompt, {
      system: options.system ?? this.defaults.system,
      ...options,
      sessionId: this.sessionId,
    });
  }

  async stream(prompt: string, options: StreamOptions = {}): Promise<string> {
    return this.llm.stream(prompt, {
      system: options.system ?? this.defaults.system,
      ...options,
      sessionId: this.sessionId,
    });
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

export {
  DeviceClient,
  probeDevice,
  DEFAULT_TEMPERATURE,
  DEFAULT_MAX_TOKENS,
  assertTools,
  parseImageFlag,
  withDocuments,
} from './device.js';
export type {
  DeviceProbe,
  DeviceRequest,
  DeviceClientOptions,
  ModelCapabilities,
  HelperFeatures,
  CloudQuota,
  BuiltInTool,
  HistoryTurn,
  ImageAttachment,
  SamplingMode,
  UseCase,
  Guardrails,
} from './device.js';
export {
  CloudClient,
  probeCloud,
  installCloudShortcut,
  shortcutDefinition,
  cloudSetupHint,
  CLOUD_SHORTCUT_NAME,
  CLOUD_SHORTCUT_NAME_WEB,
  CLOUD_CONTEXT_TOKENS,
} from './cloud.js';
export type { CloudProbe, CloudRequest } from './cloud.js';
export { toAppleSchema } from './schema.js';
export type { JsonSchema } from './schema.js';
export { parseLlmJson, stripCodeFences, extractJsonSpan } from './json-recovery.js';
export { isAppleSiliconMac, targetTripleFrom, hostTarget } from './target.js';
export { cacheDir, fingerprint, helperSource, ensureBinary } from './compile.js';
export type { Progress, OnProgress } from './compile.js';
export {
  AppleLLMError,
  ModelUnavailableError,
  SchemaRejectedError,
  ContextLengthError,
  QuotaError,
  TimeoutError,
  SetupRequiredError,
  RefusalError,
} from './errors.js';
export type { UnavailableReason } from './errors.js';
