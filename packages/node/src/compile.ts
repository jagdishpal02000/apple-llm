import { execFile } from 'node:child_process';
import { chmod, mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { ModelUnavailableError } from './errors.js';
import { findSwiftc, hostTarget } from './target.js';

const execFileAsync = promisify(execFile);

export interface Progress {
  status: string;
  percent?: number;
}
export type OnProgress = (p: Progress) => void;

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
 * A helper edit or an OS upgrade changes it, so both rebuild automatically.
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

let inFlight: Promise<string> | undefined;
let cachedKey: string | undefined;

/**
 * Compile the helper on first use and cache it. Never called at install time —
 * importing this package on Linux, an Intel Mac or macOS 25 must not fail.
 *
 * `force` bypasses the in-process memo, so a caller can rebuild after the cached
 * binary is removed or found corrupt without restarting the program.
 *
 * The memo is fingerprint-aware: a helper edit or OS upgrade changes the
 * fingerprint, so a stale in-process memo is discarded rather than reused.
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
      const source = await helperSource();
      const triple = (await hostTarget()) ?? process.arch;
      if (fingerprint(source, triple) === cachedKey) return pending;
      // Fingerprint changed (helper edited or OS upgraded): fall through
      // and rebuild rather than reuse the stale binary.
      inFlight = undefined;
      cachedKey = undefined;
    } catch {
      return pending;
    }
  }
  if (inFlight === undefined) {
    try {
      const source = await helperSource();
      const triple = (await hostTarget()) ?? process.arch;
      cachedKey = fingerprint(source, triple);
    } catch {
      // Fingerprint is best effort; build() recomputes it anyway.
    }
    inFlight = build(onProgress).catch((err) => {
      inFlight = undefined;
      cachedKey = undefined;
      throw err;
    });
  }
  return inFlight;
}

async function build(onProgress?: OnProgress): Promise<string> {
  const source = await helperSource();
  const triple = (await hostTarget()) ?? process.arch;
  const dir = cacheDir();
  const target = path.join(dir, `fm-helper-${fingerprint(source, triple)}`);

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

  onProgress?.({ status: 'building the Apple on-device helper (one time, a few seconds)' });
  await mkdir(dir, { recursive: true });
  // Random suffix: same-PID concurrent force:true builds must not collide on
  // one shared temp name.
  const unique = `${process.pid}.${Math.random().toString(36).slice(2)}`;
  const sourcePath = `${target}.${unique}.swift`;
  const staging = `${target}.${unique}.tmp`;
  const { writeFile } = await import('node:fs/promises');
  await writeFile(sourcePath, source, 'utf8');

  try {
    const targetArgs = triple.includes('-apple-') ? ['-target', triple] : [];
    await execFileAsync(swiftc.command, [
      ...swiftc.prefixArgs,
      ...targetArgs,
      '-parse-as-library',
      '-O',
      sourcePath,
      '-o',
      staging,
    ]);
    await chmod(staging, 0o755);
    // Atomic publish, so concurrent first-runs cannot observe a half-written file.
    await rename(staging, target);
  } catch (err) {
    await rm(staging, { force: true });
    // A concurrent run may have finished first, which is a success for us.
    if (await fileExists(target)) return target;
    const reason = err instanceof Error ? err.message : String(err);
    throw new ModelUnavailableError(
      `Failed to build the Apple on-device helper with ${swiftc.command}.\n${reason.slice(0, 400)}\n` +
        'This usually means the macOS SDK predates the Foundation Models framework (macOS 26+).',
      'unsupportedOSVersion',
      'device',
    );
  } finally {
    await rm(sourcePath, { force: true });
  }

  onProgress?.({ status: 'helper built' });
  return target;
}
