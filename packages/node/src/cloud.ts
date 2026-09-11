import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import type { OnProgress } from './compile.js';
import {
  AppleLLMError,
  ContextLengthError,
  QuotaError,
  SetupRequiredError,
  TimeoutError,
} from './errors.js';
import { parseLlmJson } from './json-recovery.js';
import type { JsonSchema } from './schema.js';
import { isAppleSiliconMac } from './target.js';
import type { CloudQuota } from './device.js';

const execFileAsync = promisify(execFile);

/**
 * Apple's Private Cloud Compute model, reached through the Shortcuts action
 * `is.workflow.actions.askllm`.
 *
 * Why this route rather than the framework: macOS 27 exposes
 * `FoundationModels.PrivateCloudComputeLanguageModel` as public API and it even
 * reports `isAvailable: true`, but every call fails with `ModelManagerError
 * 1046` unless the process carries `com.apple.developer.private-cloud-compute`.
 * That entitlement is AMFI-restricted — an ad-hoc-signed binary carrying it is
 * SIGKILLed (exit 137), and wrapping it in a signed .app with a real bundle ID
 * does not help. It needs a paid Developer Program provisioning profile, which
 * no installable package can ship. Shortcuts.app already holds the entitlement
 * and `/usr/bin/shortcuts` is public, so a generated shortcut is the only route
 * that works. Do not spend time re-confirming this.
 *
 * This tier sends your prompt off the machine. It is not local.
 *
 * Measured on an M4 Air: ~2s typical, ~11s at 14k tokens.
 */

/** Distinctive, so `shortcuts run` cannot match a shortcut the user wrote. */
export const CLOUD_SHORTCUT_NAME = 'Apple LLM Cloud';
/** Web search is fixed in the shortcut at install time, so it needs its own copy. */
export const CLOUD_SHORTCUT_NAME_WEB = 'Apple LLM Cloud Web';

/**
 * Hard ceiling on one call. Generous next to the ~2s typical case because the
 * failure it guards is not slowness but a *wedged* run: a misconfigured shortcut
 * whose prompt parameter did not bind pops an interactive "Request…" panel and
 * then waits for a human forever.
 */
const CALL_TIMEOUT_MS = 120_000;

/** Context window, measured empirically: 14.4k succeeds, ~33.5k is refused. */
export const CLOUD_CONTEXT_TOKENS = 32_768;

/** Import is asynchronous and unhurried (5-15s typical). */
const IMPORT_TIMEOUT_MS = 90_000;
const IMPORT_POLL_MS = 2_000;

export interface CloudProbe {
  available: boolean;
  reason?: string;
  installed?: boolean;
  contextSize?: number;
  /**
   * Real quota state, read from `PrivateCloudComputeLanguageModel.quotaUsage`
   * by the on-device helper. Absent when the helper could not run (macOS 26, or
   * no on-device model), since that is the only thing that can read it.
   */
  quota?: CloudQuota;
}

/**
 * The shortcut definition, as plain JSON.
 *
 * Every key here was confirmed against a shortcut built in the Shortcuts GUI and
 * exported, rather than guessed — the difference matters because a wrong
 * parameter name does not fail loudly. Shortcuts imports the action, silently
 * discards the unrecognised parameter, and the action then blocks on its
 * interactive prompt at run time, forever. This is the single most expensive
 * mistake available on this path.
 *
 * In particular the prompt key is `WFLLMPrompt`, *not* the `WFInput` that the
 * action's own localised strings suggest.
 *
 * The model key is deliberately absent: with no model key the action uses its
 * default, which is the Cloud (Private Cloud Compute) tier.
 */
export function shortcutDefinition(webSearch = false): Record<string, unknown> {
  const parameters: Record<string, unknown> = {
    // Stable UUID: re-running setup re-imports the same action rather than
    // accumulating variants.
    UUID: webSearch ? 'B47F1A90-2D5E-4C83-A6F1-9E0C7B34D215' : '3827CFFE-3A65-4456-BFA0-49EF061ACEE3',
    // Plain text out. "Automatic" reshapes the result to suit whatever action
    // comes next, and nothing comes next here.
    WFGenerativeResultType: 'Text',
    // The shortcut's own input, spliced in as a variable. U+FFFC is the
    // object-replacement character marking the attachment's position.
    WFLLMPrompt: {
      Value: {
        string: '￼',
        attachmentsByRange: { '{0, 1}': { Type: 'ExtensionInput' } },
      },
      WFSerializationType: 'WFTextTokenString',
    },
  };
  // "Use Broad World Knowledge" — the live-internet flag. Only set when asked,
  // so the default shortcut stays byte-identical to the confirmed-good one.
  if (webSearch) parameters.WFAllowWebSearch = true;

  return {
    WFWorkflowMinimumClientVersionString: '900',
    WFWorkflowMinimumClientVersion: 900,
    WFWorkflowClientVersion: '5037.0.17',
    WFWorkflowIcon: {
      WFWorkflowIconStartColor: 431817727,
      WFWorkflowIconGlyphNumber: 61440,
    },
    WFWorkflowOutputContentItemClasses: [],
    WFWorkflowHasOutputFallback: false,
    WFWorkflowActions: [
      {
        WFWorkflowActionIdentifier: 'is.workflow.actions.askllm',
        WFWorkflowActionParameters: parameters,
      },
    ],
    WFWorkflowInputContentItemClasses: ['WFStringContentItem', 'WFRichTextContentItem'],
    WFWorkflowImportQuestions: [],
    WFQuickActionSurfaces: [],
    WFWorkflowTypes: ['WFWorkflowTypeShowInSearch'],
    WFWorkflowHasShortcutInputVariables: true,
  };
}

async function listShortcuts(): Promise<string[]> {
  try {
    const { stdout } = await execFileAsync('shortcuts', ['list'], { timeout: 20_000 });
    return stdout.split('\n').map((l) => l.trim()).filter((l) => l !== '');
  } catch {
    return [];
  }
}

/** How many installed shortcuts carry the name. >1 breaks `shortcuts run`. */
export async function installCount(name = CLOUD_SHORTCUT_NAME): Promise<number> {
  return (await listShortcuts()).filter((l) => l === name).length;
}

export function cloudSetupHint(): string {
  return (
    'The cloud tier needs a one-time shortcut installed.\n' +
    'Run:  apple-llm setup-cloud\n' +
    'It generates and signs the shortcut on this machine — no account, no API key, nothing uploaded.'
  );
}

function duplicateMessage(count: number, name: string): string {
  return (
    `${count} shortcuts are named "${name}"; \`shortcuts run\` cannot tell them apart ` +
    'and fails with "Couldn\'t find shortcut".\n' +
    'Delete the duplicates in the Shortcuts app, then try again.'
  );
}

/**
 * Generate, sign and install the shortcut. Idempotent unless `force` is set.
 *
 * Signing needs no developer account and no signing identity: `shortcuts sign -m
 * anyone` issues a per-signature certificate that chains to Apple Root CA - G3
 * on the device. Every user signs their own copy, so nothing has to be
 * pre-signed, hosted, or shipped in the package.
 */
export async function installCloudShortcut(
  onProgress?: OnProgress,
  options: { force?: boolean; webSearch?: boolean } = {},
): Promise<void> {
  const name = options.webSearch === true ? CLOUD_SHORTCUT_NAME_WEB : CLOUD_SHORTCUT_NAME;
  const existing = await installCount(name);
  if (existing > 1) throw new AppleLLMError(duplicateMessage(existing, name), 'cloud');
  if (existing === 1 && options.force !== true) {
    onProgress?.({ status: `"${name}" is already installed` });
    return;
  }

  const dir = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-shortcut-'));
  try {
    // plutil converts JSON to a plist, which saves hand-rolling plist XML.
    // `shortcuts sign` insists on a .shortcut extension for its input.
    const jsonPath = path.join(dir, 'definition.json');
    const unsignedPath = path.join(dir, 'unsigned.shortcut');
    const signedPath = path.join(dir, `${name}.shortcut`);

    await writeFile(jsonPath, JSON.stringify(shortcutDefinition(options.webSearch)), 'utf8');
    await execFileAsync('plutil', ['-convert', 'binary1', '-o', unsignedPath, jsonPath]);

    onProgress?.({ status: 'signing the shortcut locally' });
    await execFileAsync('shortcuts', ['sign', '-m', 'anyone', '-i', unsignedPath, '-o', signedPath]);

    // `open` hands the file to Shortcuts, which imports it in the background.
    onProgress?.({ status: `installing "${name}" (Shortcuts imports asynchronously)` });
    await execFileAsync('open', [signedPath]);

    const deadline = Date.now() + IMPORT_TIMEOUT_MS;
    for (;;) {
      if ((await installCount(name)) >= 1) break;
      if (Date.now() > deadline) {
        throw new SetupRequiredError(
          `Shortcuts did not import "${name}" within ${IMPORT_TIMEOUT_MS / 1000}s.\n` +
            'If a confirmation panel is open in the Shortcuts app, accept it and run setup again.',
          'setup-cloud',
          'cloud',
        );
      }
      await new Promise((resolve) => setTimeout(resolve, IMPORT_POLL_MS));
    }
    onProgress?.({ status: `"${name}" installed` });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/** Probe the cloud tier without constructing a client. */
export async function probeCloud(): Promise<CloudProbe> {
  if (process.platform !== 'darwin') {
    return { available: false, installed: false, reason: `unsupportedPlatform: this machine reports "${process.platform}"` };
  }
  if (!isAppleSiliconMac()) {
    return { available: false, installed: false, reason: 'deviceNotEligible: Apple Intelligence needs Apple Silicon' };
  }
  try {
    await execFileAsync('shortcuts', ['list'], { timeout: 20_000 });
  } catch {
    return { available: false, installed: false, reason: 'this system does not provide /usr/bin/shortcuts' };
  }
  const count = await installCount();
  if (count === 0) {
    return { available: false, installed: false, reason: cloudSetupHint() };
  }
  if (count > 1) {
    return { available: false, installed: true, reason: duplicateMessage(count, CLOUD_SHORTCUT_NAME) };
  }
  return { available: true, installed: true, contextSize: CLOUD_CONTEXT_TOKENS };
}

export interface CloudRequest {
  system?: string;
  prompt: string;
  /** Run with "Use Broad World Knowledge" — needs the web shortcut installed. */
  webSearch?: boolean;
}

/**
 * The Private Cloud Compute tier. Free but quota'd, and text-out only: there is
 * no constrained decoding here, so `json()` asks for JSON in the prompt and
 * recovers it from prose rather than guaranteeing it.
 */
export class CloudClient {
  private ready = false;
  /** Last known quota, set by `probe()` so a call can fail fast. */
  private quota?: CloudQuota;

  /**
   * Tell the client what the framework reported about the quota.
   *
   * Worth doing because a `shortcuts run` against an exhausted quota costs a
   * full round trip to find out; this turns that into an immediate typed error.
   */
  setQuota(quota: CloudQuota | undefined): void {
    this.quota = quota;
  }

  private assertQuota(): void {
    if (this.quota?.status !== 'limitReached') return;
    throw new QuotaError(
      'Apple Private Cloud Compute quota is exhausted (reported by the framework ' +
        'before the call was made).\nFall back to the on-device tier, or try again later.',
      'cloud',
      this.quota.resetDate === undefined ? undefined : new Date(this.quota.resetDate),
    );
  }

  get label(): string {
    return 'apple private cloud compute';
  }

  get contextSize(): number {
    return CLOUD_CONTEXT_TOKENS;
  }

  async ensureReady(onProgress?: OnProgress): Promise<void> {
    if (this.ready) return;
    const probe = await probeCloud();
    if (!probe.available) {
      if (probe.installed !== true) {
        throw new SetupRequiredError(probe.reason ?? cloudSetupHint(), 'setup-cloud', 'cloud');
      }
      throw new AppleLLMError(probe.reason ?? 'the cloud tier is unavailable', 'cloud');
    }
    this.ready = true;
    onProgress?.({ status: `apple private cloud compute ready (${CLOUD_CONTEXT_TOKENS} token context)` });
  }

  async text(request: CloudRequest): Promise<string> {
    await this.ensureReady();
    this.assertQuota();
    const name = request.webSearch === true ? CLOUD_SHORTCUT_NAME_WEB : CLOUD_SHORTCUT_NAME;
    if (request.webSearch === true && (await installCount(name)) !== 1) {
      throw new SetupRequiredError(
        `webSearch needs the "${name}" shortcut.\nRun:  apple-llm setup-cloud --web-search`,
        'setup-cloud --web-search',
        'cloud',
      );
    }
    const prompt = request.system ? `${request.system}\n\n${request.prompt}` : request.prompt;

    const dir = await mkdtemp(path.join(os.tmpdir(), 'apple-llm-cloud-'));
    try {
      const inPath = path.join(dir, 'prompt.txt');
      const outPath = path.join(dir, 'reply.txt');
      await writeFile(inPath, prompt, 'utf8');

      try {
        await execFileAsync('shortcuts', ['run', name, '-i', inPath, '-o', outPath], {
          timeout: CALL_TIMEOUT_MS,
          killSignal: 'SIGKILL',
        });
      } catch (error) {
        throw describeRunFailure(error, name);
      }

      let reply: string;
      try {
        reply = await readFile(outPath, 'utf8');
      } catch {
        throw new AppleLLMError('Apple cloud generation produced no output.', 'cloud');
      }
      if (reply.trim() === '') {
        throw new AppleLLMError('Apple cloud generation returned an empty reply.', 'cloud');
      }
      return reply;
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  /**
   * There is no constrained decoding on this tier, so the schema is spelled out
   * in the prompt and the reply is mined for JSON. The shape is a request here,
   * not a guarantee — unlike on device.
   */
  async json(request: CloudRequest & { schema: JsonSchema }): Promise<unknown> {
    const instruction =
      'Reply with a single JSON value matching this JSON Schema. ' +
      'Output only the JSON, with no commentary and no code fence.\n\n' +
      `${JSON.stringify(request.schema, null, 2)}`;
    const system = request.system ? `${request.system}\n\n${instruction}` : instruction;
    return parseLlmJson(await this.text({ ...request, system }));
  }

  // eslint-disable-next-line class-methods-use-this
  close(): void {
    /* nothing long-lived: each call is its own `shortcuts run`. */
  }
}

/**
 * Turn a `shortcuts run` rejection into a typed error. The failures worth naming
 * are the context ceiling, the quota, and a timeout — which almost always means
 * the shortcut is prompting for input rather than running unattended.
 */
export function describeRunFailure(error: unknown, name = CLOUD_SHORTCUT_NAME): Error {
  const err = error as { killed?: boolean; stderr?: string; message?: string };
  if (err.killed === true) {
    return new TimeoutError(
      `Apple cloud generation got no response within ${CALL_TIMEOUT_MS / 1000}s. ` +
        `Open "${name}" in the Shortcuts app and check that Request is bound to the ` +
        'Shortcut Input variable — an unbound Request makes the action wait for a human.',
      'cloud',
    );
  }
  const stderr = err.stderr?.trim() ?? '';
  if (/maximum allowed length/i.test(stderr)) {
    return new ContextLengthError(
      `The prompt exceeded the cloud model's ~${CLOUD_CONTEXT_TOKENS}-token context window.`,
      'cloud',
    );
  }
  if (/QuotaLimitReached|quota/i.test(stderr)) {
    // resetDate appears in the message as an ISO-ish timestamp when present.
    const match = /(\d{4}-\d{2}-\d{2}[T ][\d:]+)/.exec(stderr);
    const resetDate = match ? new Date(match[1].replace(' ', 'T')) : undefined;
    return new QuotaError(
      `Apple Private Cloud Compute quota reached.${resetDate ? ` Resets ${resetDate.toISOString()}.` : ''}\n` +
        'Fall back to the on-device tier, or try again later.',
      'cloud',
      resetDate !== undefined && !Number.isNaN(resetDate.getTime()) ? resetDate : undefined,
    );
  }
  if (/Couldn.t find shortcut/i.test(stderr)) {
    return new SetupRequiredError(
      `Shortcuts could not find "${name}".\n${cloudSetupHint()}`,
      'setup-cloud',
      'cloud',
    );
  }
  return new AppleLLMError(
    `Apple cloud generation failed: ${stderr !== '' ? stderr : (err.message ?? String(error))}`,
    'cloud',
  );
}
