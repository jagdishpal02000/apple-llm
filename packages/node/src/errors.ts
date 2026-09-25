/**
 * Typed errors, so callers branch on a class rather than matching a message.
 * Apple rewords its diagnostics between OS releases; the helper classifies
 * failures at the Swift boundary and this module names them.
 *
 * Every error also carries a stable `code`, for code that would rather
 * `switch` than chain `instanceof`, and for logs.
 */

export type Tier = 'device' | 'cloud';

export type ErrorCode =
  | 'APPLE_LLM_ERROR'
  | 'MODEL_UNAVAILABLE'
  | 'SCHEMA_REJECTED'
  | 'SCHEMA_VALIDATION'
  | 'CONTEXT_LENGTH'
  | 'QUOTA'
  | 'TIMEOUT'
  | 'ABORTED'
  | 'SETUP_REQUIRED'
  | 'REFUSAL'
  | 'TOOL_EXECUTION'
  | 'MODEL_BUSY'
  | 'UNSUPPORTED';

export interface AppleLLMErrorOptions {
  cause?: unknown;
}

export class AppleLLMError extends Error {
  readonly code: ErrorCode = 'APPLE_LLM_ERROR';

  constructor(message: string, readonly tier?: Tier, options?: AppleLLMErrorOptions) {
    super(message, options?.cause === undefined ? undefined : { cause: options.cause });
    this.name = new.target.name;
  }
}

/** Why the on-device model cannot be used. Distinguishable, because the fix differs. */
export type UnavailableReason =
  | 'appleIntelligenceNotEnabled'
  | 'modelNotReady'
  | 'deviceNotEligible'
  | 'unsupportedPlatform'
  | 'unsupportedOSVersion'
  | 'noSwiftCompiler'
  | 'unknown';

export class ModelUnavailableError extends AppleLLMError {
  override readonly code = 'MODEL_UNAVAILABLE';

  constructor(message: string, readonly reason: UnavailableReason, tier?: Tier, options?: AppleLLMErrorOptions) {
    super(message, tier, options);
  }
}

/** Apple's `GenerationSchema` decoder refused the schema. Almost always a dialect rule. */
export class SchemaRejectedError extends AppleLLMError {
  override readonly code = 'SCHEMA_REJECTED';
}

/**
 * The reply parsed, but your schema library rejected it — a refinement the
 * decoder cannot enforce (`.email()`, `.min(3)`), or, on the cloud tier, a
 * shape the model got wrong. `text` is what the model actually said.
 */
export class SchemaValidationError extends AppleLLMError {
  override readonly code = 'SCHEMA_VALIDATION';

  constructor(
    message: string,
    readonly issues: ReadonlyArray<{ message: string; path?: ReadonlyArray<unknown> }>,
    readonly text: string,
    tier?: Tier,
  ) {
    super(message, tier);
  }
}

/** The prompt (plus transcript) exceeded the model's context window. */
export class ContextLengthError extends AppleLLMError {
  override readonly code = 'CONTEXT_LENGTH';

  constructor(message: string, tier?: Tier, readonly contextSize?: number, readonly tokenCount?: number) {
    super(message, tier);
  }
}

/** Private Cloud Compute is rate limited, or the on-device model reported `rateLimited`. */
export class QuotaError extends AppleLLMError {
  override readonly code = 'QUOTA';

  constructor(message: string, tier?: Tier, readonly resetDate?: Date) {
    super(message, tier);
  }
}

export class TimeoutError extends AppleLLMError {
  override readonly code = 'TIMEOUT';
}

/**
 * The caller's `AbortSignal` fired. `name` is `'AbortError'`, matching
 * `fetch`, so existing `err.name === 'AbortError'` checks keep working.
 */
export class AbortError extends AppleLLMError {
  override readonly code = 'ABORTED';
}

/** A one-time setup step has not been run — currently only the cloud shortcut. */
export class SetupRequiredError extends AppleLLMError {
  override readonly code = 'SETUP_REQUIRED';

  constructor(message: string, readonly step: string, tier?: Tier) {
    super(message, tier);
  }
}

/** The model declined to answer (guardrail or refusal). */
export class RefusalError extends AppleLLMError {
  override readonly code = 'REFUSAL';
}

/**
 * One of your function tools threw. The original error is `cause`, untouched,
 * so a bug in a tool reads as a bug in the tool — not as a model failure.
 */
export class ToolExecutionError extends AppleLLMError {
  override readonly code = 'TOOL_EXECUTION';

  constructor(message: string, readonly toolName: string, options?: AppleLLMErrorOptions) {
    super(message, 'device', options);
  }
}

/**
 * Apple's model manager turned the request away because other processes are
 * using the model (ModelManagerError 1042). Already retried with backoff
 * before this is raised; retrying later usually works.
 */
export class ModelBusyError extends AppleLLMError {
  override readonly code = 'MODEL_BUSY';
}

/** A capability this tier, OS or model does not have. */
export class UnsupportedError extends AppleLLMError {
  override readonly code = 'UNSUPPORTED';
}

/** Turn an unavailability reason into something the user can act on. */
export function unavailableMessage(reason?: string): string {
  const base = 'Apple’s on-device model is not available';
  if (reason?.includes('appleIntelligenceNotEnabled')) {
    return `${base}: Apple Intelligence is turned off.\nEnable it in System Settings › Apple Intelligence & Siri, then try again.`;
  }
  if (reason?.includes('modelNotReady')) {
    return `${base}: the model is still downloading.\nLeave the Mac online and plugged in for a few minutes, then try again.`;
  }
  if (reason?.includes('deviceNotEligible')) {
    return `${base}: this Mac is not eligible for Apple Intelligence.`;
  }
  if (reason?.includes('unsupportedPlatform')) {
    return `${base}: this platform is not supported (macOS on Apple Silicon only).`;
  }
  if (reason?.includes('unsupportedOSVersion')) {
    return `${base}: needs macOS 26 or later.`;
  }
  if (reason?.includes('noSwiftCompiler')) {
    return (
      `${base}: no Swift compiler found.\n` +
      'Install the Xcode command line tools with `xcode-select --install`, then try again.'
    );
  }
  return `${base}${reason ? `: ${reason}` : '.'}`;
}

/** Narrow a helper `reason` string to the typed union. */
export function toUnavailableReason(reason?: string): UnavailableReason {
  if (reason?.includes('appleIntelligenceNotEnabled')) return 'appleIntelligenceNotEnabled';
  if (reason?.includes('modelNotReady')) return 'modelNotReady';
  if (reason?.includes('deviceNotEligible')) return 'deviceNotEligible';
  if (reason?.includes('unsupportedPlatform')) return 'unsupportedPlatform';
  if (reason?.includes('unsupportedOSVersion')) return 'unsupportedOSVersion';
  if (reason?.includes('noSwiftCompiler')) return 'noSwiftCompiler';
  return 'unknown';
}
