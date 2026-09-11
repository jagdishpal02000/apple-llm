/**
 * Typed errors, so callers branch on a class rather than matching a message.
 * Apple rewords its diagnostics between OS releases; the helper classifies
 * failures at the Swift boundary and this module names them.
 */

export type Tier = 'device' | 'cloud';

export class AppleLLMError extends Error {
  constructor(message: string, readonly tier?: Tier) {
    super(message);
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
  constructor(message: string, readonly reason: UnavailableReason, tier?: Tier) {
    super(message, tier);
  }
}

/** Apple's `GenerationSchema` decoder refused the schema. Almost always a dialect rule. */
export class SchemaRejectedError extends AppleLLMError {}

/** The prompt (plus transcript) exceeded the model's context window. */
export class ContextLengthError extends AppleLLMError {
  constructor(message: string, tier?: Tier, readonly contextSize?: number) {
    super(message, tier);
  }
}

/** Private Cloud Compute is rate limited, or the on-device model reported `rateLimited`. */
export class QuotaError extends AppleLLMError {
  constructor(message: string, tier?: Tier, readonly resetDate?: Date) {
    super(message, tier);
  }
}

export class TimeoutError extends AppleLLMError {}

/** A one-time setup step has not been run — currently only the cloud shortcut. */
export class SetupRequiredError extends AppleLLMError {
  constructor(message: string, readonly step: string, tier?: Tier) {
    super(message, tier);
  }
}

/** The model declined to answer (guardrail or refusal). */
export class RefusalError extends AppleLLMError {}

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
