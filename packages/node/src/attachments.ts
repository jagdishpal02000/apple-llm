/**
 * Images as the helper wants them: files on disk.
 *
 * Apple's `Attachment(imageURL:)` reads a file URL, while the AI SDK and
 * OpenAI-style clients send bytes, base64, data URLs or http URLs. Each is
 * written to a private temp directory for the duration of one call and removed
 * afterwards.
 */
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { AppleLLMError } from './errors.js';

/** Largest image accepted from a URL or inline data. */
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

const EXTENSIONS: Record<string, string> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/jpg': '.jpg',
  'image/gif': '.gif',
  'image/webp': '.webp',
  'image/heic': '.heic',
  'image/heif': '.heif',
  'image/tiff': '.tiff',
  'image/bmp': '.bmp',
};

/** Image data in any form a client might send. */
export type ImageSource = Uint8Array | ArrayBuffer | URL | string;

/** A scratch directory whose files disappear with `dispose()`. */
export class TempImages {
  private dir: string | undefined;
  private count = 0;

  /** Write one image and return its path. */
  async add(source: ImageSource, mediaType?: string, signal?: AbortSignal): Promise<string> {
    const { bytes, type } = await loadImage(source, mediaType, signal);
    this.dir ??= await mkdtemp(path.join(os.tmpdir(), 'apple-llm-img-'));
    this.count += 1;
    const file = path.join(this.dir, `image${this.count}${EXTENSIONS[type] ?? '.img'}`);
    await writeFile(file, bytes);
    return file;
  }

  async dispose(): Promise<void> {
    if (this.dir !== undefined) await rm(this.dir, { recursive: true, force: true }).catch(() => undefined);
    this.dir = undefined;
  }
}

async function loadImage(
  source: ImageSource,
  mediaType: string | undefined,
  signal?: AbortSignal,
): Promise<{ bytes: Uint8Array; type: string }> {
  if (source instanceof Uint8Array) return { bytes: checked(source), type: mediaType ?? sniff(source) };
  if (source instanceof ArrayBuffer) {
    const bytes = new Uint8Array(source);
    return { bytes: checked(bytes), type: mediaType ?? sniff(bytes) };
  }
  const text = source instanceof URL ? source.href : source;
  const dataUrl = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(text);
  if (dataUrl !== null) {
    const bytes = dataUrl[2] !== undefined
      ? Buffer.from(dataUrl[3], 'base64')
      : Buffer.from(decodeURIComponent(dataUrl[3]), 'utf8');
    return { bytes: checked(bytes), type: dataUrl[1] ?? mediaType ?? sniff(bytes) };
  }
  if (/^https?:\/\//i.test(text)) {
    const response = await fetch(text, { signal });
    if (!response.ok) throw new AppleLLMError(`Could not download image ${text}: HTTP ${response.status}`);
    const declared = Number(response.headers.get('content-length') ?? '0');
    if (declared > MAX_IMAGE_BYTES) throw tooLarge();
    const bytes = new Uint8Array(await response.arrayBuffer());
    const type = response.headers.get('content-type')?.split(';')[0]?.trim();
    return { bytes: checked(bytes), type: mediaType ?? type ?? sniff(bytes) };
  }
  // Anything else is base64 without a data: prefix, as the AI SDK sends it.
  const bytes = Buffer.from(text, 'base64');
  if (bytes.length === 0) throw new AppleLLMError('Image data is empty or not base64.');
  return { bytes: checked(bytes), type: mediaType ?? sniff(bytes) };
}

function checked(bytes: Uint8Array): Uint8Array {
  if (bytes.length > MAX_IMAGE_BYTES) throw tooLarge();
  return bytes;
}

function tooLarge(): AppleLLMError {
  return new AppleLLMError(`Image is larger than ${MAX_IMAGE_BYTES / 1024 / 1024} MB.`);
}

/** Media type from magic bytes, for sources that do not say. */
function sniff(bytes: Uint8Array): string {
  const at = (i: number): number => bytes[i] ?? -1;
  if (at(0) === 0x89 && at(1) === 0x50 && at(2) === 0x4e && at(3) === 0x47) return 'image/png';
  if (at(0) === 0xff && at(1) === 0xd8) return 'image/jpeg';
  if (at(0) === 0x47 && at(1) === 0x49 && at(2) === 0x46) return 'image/gif';
  if (at(8) === 0x57 && at(9) === 0x45 && at(10) === 0x42 && at(11) === 0x50) return 'image/webp';
  return 'application/octet-stream';
}
