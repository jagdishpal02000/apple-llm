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
 * Build the compile target from a version string, e.g. `27.0` ->
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

/** `major.minor` as two numbers, or null when the text is not a version. */
function versionParts(text: string | null): [number, number] | null {
  if (text === null) return null;
  const match = /^(\d+)(?:\.(\d+))?/.exec(text.trim());
  if (match === null) return null;
  return [Number(match[1]), Number(match[2] ?? '0')];
}

/**
 * The version to build for: the older of the SDK and the running OS.
 *
 * The SDK alone is not enough. A Mac on macOS 26 can have the macOS 27 SDK, and
 * a binary built for 27 does not run on 26; a Mac on macOS 27 can still have
 * the macOS 26 SDK, which cannot target an OS it predates. The helper checks
 * every newer API at run time, so building for the older of the two is always
 * safe. Pure, so it can be tested on any machine; the Python package has the
 * same rule, and the two must agree to share one compiled helper.
 */
export function deploymentVersion(sdk: string | null, os: string | null): string | null {
  const a = versionParts(sdk);
  const b = versionParts(os);
  const pick = a === null ? b : b === null ? a : a[0] !== b[0] ? (a[0] < b[0] ? a : b) : a[1] <= b[1] ? a : b;
  return pick === null ? null : `${pick[0]}.${pick[1]}`;
}

async function firstLine(cmd: string, args: string[]): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync(cmd, args);
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

/** The SDK version the Swift compiler would build against, or null. */
export function sdkVersion(): Promise<string | null> {
  return firstLine('xcrun', ['--show-sdk-version']);
}

/** The compile target: the older of the SDK and the running OS (see deploymentVersion). */
export async function hostTarget(): Promise<string | null> {
  const [sdk, os] = await Promise.all([sdkVersion(), firstLine('sw_vers', ['-productVersion'])]);
  const version = deploymentVersion(sdk, os);
  return version === null ? null : targetTripleFrom(version);
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
