import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

/**
 * True when this machine could plausibly run Apple's on-device model: Apple
 * Silicon on macOS. Availability itself is confirmed by probing the helper.
 */
export function isAppleSiliconMac(): boolean {
  return process.platform === 'darwin' && process.arch === 'arm64';
}

/**
 * Build the compile target from an SDK version string, e.g. `27.0` ->
 * `arm64-apple-macos27.0`.
 *
 * swiftc's own default triple carries a patch component
 * (`arm64-apple-macosx27.0.0`) that no shipped stdlib matches, so the target has
 * to be spelled out. Pure so it can be tested against captured `xcrun` output on
 * any machine.
 */
export function targetTripleFrom(versionOutput: string, arch: string = process.arch): string | null {
  const [major, minor = '0'] = versionOutput.trim().split('.');
  if (!major || !/^\d+$/.test(major)) return null;
  if (minor !== undefined && !/^\d+$/.test(minor)) return null;
  // Normalize so an npm and a pip user on one machine derive the same triple:
  // Node says arm64/x64, Python says arm64/x86_64. Any non-x64 arch falls back
  // to arm64, matching the Python mapping, so both agree on one cached binary.
  const cpu = arch === 'x64' || arch === 'x86_64' ? 'x86_64' : 'arm64';
  return `${cpu}-apple-macos${major}.${minor}`;
}

/** The major macOS version, or null if it cannot be read. */
export async function macosMajor(): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync('sw_vers', ['-productVersion']);
    const major = Number.parseInt(stdout.trim().split('.')[0] ?? '', 10);
    return Number.isFinite(major) ? major : null;
  } catch {
    return null;
  }
}

/** Ask the SDK, then the OS, for a version to build the triple from. */
export async function hostTarget(): Promise<string | null> {
  for (const [cmd, args] of [
    ['xcrun', ['--show-sdk-version']],
    ['sw_vers', ['-productVersion']],
  ] as Array<[string, string[]]>) {
    try {
      const { stdout } = await execFileAsync(cmd, args);
      const triple = targetTripleFrom(stdout);
      if (triple !== null) return triple;
    } catch {
      // try the next source
    }
  }
  return null;
}

export interface SwiftCompiler {
  command: string;
  prefixArgs: string[];
}

/**
 * Locate a Swift compiler. It must be invoked through `xcrun` or the
 * `/usr/bin/swiftc` shim: those set up SDKROOT, whereas the raw binary that
 * `xcrun -f swiftc` prints cannot find the standard library on its own.
 */
export async function findSwiftc(): Promise<SwiftCompiler | null> {
  try {
    await execFileAsync('xcrun', ['-f', 'swiftc']);
    return { command: 'xcrun', prefixArgs: ['swiftc'] };
  } catch {
    // fall through to the shim
  }
  const { stat } = await import('node:fs/promises');
  try {
    await stat('/usr/bin/swiftc');
    return { command: '/usr/bin/swiftc', prefixArgs: [] };
  } catch {
    return null;
  }
}
