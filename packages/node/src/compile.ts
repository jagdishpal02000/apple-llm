import { execFile } from 'node:child_process';
import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { ModelUnavailableError } from './errors.js';
import { findSwiftc, hostTarget, sdkVersion } from './target.js';

const execFileAsync = promisify(execFile);

export interface Progress {
  status: string;
  percent?: number;
}
export type OnProgress = (p: Progress) => void;

/** Where to report a helper that will not build. */
const ISSUES_URL = 'https://github.com/jagdishpal02000/apple-llm/issues';

/** How often a build that is still running says so. A cold first build takes about a minute. */
const BUILD_HEARTBEAT_MS = 15_000;

/**
 * Where the compiled helper is cached.
 *
 * Deliberately not namespaced per language binding: the npm and the pip package
 * embed byte-identical Swift, so they derive the same fingerprint and share one
 * compiled binary. Installing both costs one compile, not two.
 */
export function cacheDir(): string {
  return path.join(os.homedir(), 'Library', 'Caches', 'apple-llm', 'bin');
}

/**
 * The cache key: sha256 over the Swift source, a newline, and the target triple,
 * truncated to 12 hex characters.
 *
 * The exact recipe is load-bearing — the Python package computes the same string
 * and must agree byte for byte, or the two would each compile their own copy.
 * A helper edit, an OS upgrade or a new SDK changes it, so both rebuild
 * automatically.
 */
export function fingerprint(source: string, triple: string): string {
  return createHash('sha256').update(`${source}\n${triple}`, 'utf8').digest('hex').slice(0, 12);
}

let cachedSource: string | undefined;

/** The embedded Swift source, shipped as package data beside the built output. */
export async function helperSource(): Promise<string> {
  if (cachedSource !== undefined) return cachedSource;
  const here = path.dirname(fileURLToPath(import.meta.url));
  // dist/index.js -> ../swift/helper.swift; src/compile.ts -> ../swift/helper.swift
  const candidates = [
    path.join(here, '..', 'swift', 'helper.swift'),
    path.join(here, 'swift', 'helper.swift'),
  ];
  for (const candidate of candidates) {
    try {
      cachedSource = await readFile(candidate, 'utf8');
      return cachedSource;
    } catch {
      // try the next location
    }
  }
  throw new Error(
    `apple-llm could not find its embedded helper.swift (looked in ${candidates.join(', ')}).`,
  );
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

/** What a build is made from, and the cache key that names its result. */
interface BuildInputs {
  source: string;
  triple: string;
  key: string;
}

async function buildInputs(): Promise<BuildInputs> {
  const source = await helperSource();
  // Empty when neither the SDK nor the OS version can be read, exactly as the
  // Python package does, so the two still agree on the key. No -target is
  // passed then.
  const triple = (await hostTarget()) ?? '';
  return { source, triple, key: fingerprint(source, triple) };
}

let inFlight: Promise<string> | undefined;
let cachedKey: string | undefined;

/**
 * Compile the helper on first use and cache it. Never called at install time —
 * importing this package on Linux, an Intel Mac or macOS 25 must not fail.
 *
 * `force` bypasses the in-process memo, so a caller can rebuild after the cached
 * binary is removed or found corrupt without restarting the program.
 *
 * The memo is fingerprint-aware: a helper edit, OS upgrade or new SDK changes
 * the fingerprint, so a stale in-process memo is discarded rather than reused.
 * Concurrent first calls share one build: a cold one takes about a minute, and
 * two at once would only slow each other down.
 */
export async function ensureBinary(
  onProgress?: OnProgress,
  options: { force?: boolean } = {},
): Promise<string> {
  if (options.force === true) {
    inFlight = undefined;
    cachedKey = undefined;
  } else if (inFlight !== undefined && cachedKey !== undefined) {
    const pending = inFlight;
    try {
      if ((await buildInputs()).key === cachedKey) return pending;
      // Fingerprint changed (helper edited, OS upgraded or SDK updated): fall
      // through and rebuild rather than reuse the stale binary.
      if (inFlight === pending) {
        inFlight = undefined;
        cachedKey = undefined;
      }
    } catch {
      return pending;
    }
  }
  if (inFlight === undefined) {
    // Claimed before the first await, so a second caller arriving meanwhile
    // waits for this build instead of starting its own.
    let attempt!: Promise<string>;
    attempt = (async () => {
      const inputs = await buildInputs();
      if (inFlight === attempt) cachedKey = inputs.key;
      return build(inputs, onProgress);
    })();
    inFlight = attempt;
    attempt.catch(() => {
      if (inFlight === attempt) {
        inFlight = undefined;
        cachedKey = undefined;
      }
    });
  }
  return inFlight;
}

/**
 * The compiler's `error:` lines, which say what went wrong. swiftc's full
 * output leads with the command line and dozens of deprecation warnings, so a
 * fixed-length prefix of it usually cut off before the first error.
 */
export function compilerErrors(output: string): string {
  const errors = output
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => /(^|\s)error: /.test(line));
  if (errors.length > 0) return [...new Set(errors)].slice(0, 6).join('\n');
  return output.trim().slice(-600);
}

async function build({ source, triple, key }: BuildInputs, onProgress?: OnProgress): Promise<string> {
  const dir = cacheDir();
  const target = path.join(dir, `fm-helper-${key}`);

  if (await fileExists(target)) return target;

  const swiftc = await findSwiftc();
  if (swiftc === null) {
    throw new ModelUnavailableError(
      'apple-llm needs a Swift compiler to build its on-device helper, but none was found.\n' +
        'Install the Xcode command line tools with `xcode-select --install`, then try again.',
      'noSwiftCompiler',
      'device',
    );
  }

  await mkdir(dir, { recursive: true });
  // Random suffix: same-PID concurrent force:true builds must not collide on
  // one shared temp name.
  const unique = `${process.pid}.${Math.random().toString(36).slice(2)}`;
  const sourcePath = `${target}.${unique}.swift`;
  const staging = `${target}.${unique}.tmp`;
  await writeFile(sourcePath, source, 'utf8');

  // A first build takes about a minute on a cold compiler cache, and a silent
  // minute reads as a hang. Say so up front, then keep saying it is working.
  const started = Date.now();
  const seconds = (): number => Math.round((Date.now() - started) / 1000);
  onProgress?.({ status: 'building the Apple on-device helper (first run only; up to a minute)' });
  const heartbeat = setInterval(() => {
    onProgress?.({ status: `still building the helper (${seconds()}s)` });
  }, BUILD_HEARTBEAT_MS);
  heartbeat.unref();

  try {
    const targetArgs = triple.includes('-apple-') ? ['-target', triple] : [];
    await execFileAsync(
      swiftc.command,
      [...swiftc.prefixArgs, ...targetArgs, '-parse-as-library', '-O', sourcePath, '-o', staging],
      // Warnings alone run to tens of KB; never let output size fail a build.
      { maxBuffer: 64 * 1024 * 1024 },
    );
    await chmod(staging, 0o755);
    // Atomic publish, so concurrent first-runs cannot observe a half-written file.
    await rename(staging, target);
  } catch (err) {
    await rm(staging, { force: true });
    // A concurrent run may have finished first, which is a success for us.
    if (await fileExists(target)) return target;
    const output = (err as { stderr?: unknown }).stderr;
    const detail = compilerErrors(typeof output === 'string' && output !== '' ? output : String(err));
    const sdk = await sdkVersion();
    throw new ModelUnavailableError(
      `Failed to build the Apple on-device helper with ${swiftc.command} ` +
        `(SDK ${sdk ?? 'unknown'}, target ${triple || 'default'}).\n${detail}\n` +
        'It needs macOS 26 or later and the Xcode Command Line Tools to match. Updating them usually fixes this: ' +
        'System Settings › General › Software Update, or `xcode-select --install`.\n' +
        `If it still fails, please report it, with this message, at ${ISSUES_URL}`,
      'unsupportedOSVersion',
      'device',
    );
  } finally {
    clearInterval(heartbeat);
    await rm(sourcePath, { force: true });
  }

  onProgress?.({ status: `helper built in ${seconds()}s` });
  return target;
}
