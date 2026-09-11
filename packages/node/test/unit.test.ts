import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { shortcutDefinition, CLOUD_SHORTCUT_NAME } from '../src/cloud.js';
import { assertTools, parseImageFlag, withDocuments } from '../src/device.js';
import { extractJsonSpan, parseLlmJson, stripCodeFences } from '../src/json-recovery.js';
import { targetTripleFrom } from '../src/target.js';
import { fingerprint } from '../src/compile.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..', '..');

describe('shortcut definition', () => {
  const def = shortcutDefinition() as any;
  const params = def.WFWorkflowActions[0].WFWorkflowActionParameters;

  it('uses WFLLMPrompt, not WFInput', () => {
    // The single most expensive mistake on this path. A wrong key imports fine
    // and is then silently discarded, after which the action blocks forever on
    // an interactive "Request…" panel.
    expect(params.WFLLMPrompt).toBeDefined();
    expect(params.WFInput).toBeUndefined();
    expect(JSON.stringify(def)).not.toContain('WFInput');
  });

  it('omits the model key so the action defaults to the Cloud tier', () => {
    expect(Object.keys(params).some((k) => /model/i.test(k))).toBe(false);
  });

  it('asks for plain text out', () => {
    // "Automatic" reshapes the result for whatever action follows.
    expect(params.WFGenerativeResultType).toBe('Text');
  });

  it('splices the shortcut input in with a U+FFFC attachment', () => {
    expect(params.WFLLMPrompt.WFSerializationType).toBe('WFTextTokenString');
    expect(params.WFLLMPrompt.Value.string).toBe('￼');
    expect(params.WFLLMPrompt.Value.attachmentsByRange).toEqual({
      '{0, 1}': { Type: 'ExtensionInput' },
    });
  });

  it('targets the ask-llm action', () => {
    expect(def.WFWorkflowActions[0].WFWorkflowActionIdentifier).toBe('is.workflow.actions.askllm');
  });

  it('only sets WFAllowWebSearch when web search was asked for', () => {
    expect(params.WFAllowWebSearch).toBeUndefined();
    const web = shortcutDefinition(true) as any;
    expect(web.WFWorkflowActions[0].WFWorkflowActionParameters.WFAllowWebSearch).toBe(true);
    // The two variants must not collide in the Shortcuts library.
    expect(web.WFWorkflowActions[0].WFWorkflowActionParameters.UUID).not.toBe(params.UUID);
  });

  it('has a distinctive name so `shortcuts run` cannot match a user shortcut', () => {
    expect(CLOUD_SHORTCUT_NAME).toMatch(/Apple LLM/);
  });
});

describe('target triple derivation', () => {
  it('drops the patch component swiftc would otherwise add', () => {
    // swiftc's default host triple is arm64-apple-macosx27.0.0, which no
    // shipped stdlib matches.
    expect(targetTripleFrom('27.0\n', 'arm64')).toBe('arm64-apple-macos27.0');
  });

  it('handles a bare major version', () => {
    expect(targetTripleFrom('26', 'arm64')).toBe('arm64-apple-macos26.0');
  });

  it('ignores a patch component in the SDK output', () => {
    expect(targetTripleFrom('27.1.2', 'arm64')).toBe('arm64-apple-macos27.1');
  });

  it('normalises arch names so npm and pip agree on one binary', () => {
    // Node says x64, Python says x86_64; they must derive the same triple or
    // each would compile its own copy of the helper.
    expect(targetTripleFrom('27.0', 'x64')).toBe(targetTripleFrom('27.0', 'x86_64'));
  });

  it('returns null for unusable output', () => {
    expect(targetTripleFrom('', 'arm64')).toBeNull();
    expect(targetTripleFrom('not-a-version', 'arm64')).toBeNull();
  });
});

describe('cache fingerprint', () => {
  it('is sha256(source + newline + triple), first 12 hex chars', () => {
    // Pinned exactly: the Python package computes the same string, and the two
    // share one compiled binary only if they agree byte for byte.
    const expected = createHash('sha256')
      .update('SOURCE\narm64-apple-macos27.0', 'utf8')
      .digest('hex')
      .slice(0, 12);
    expect(fingerprint('SOURCE', 'arm64-apple-macos27.0')).toBe(expected);
    expect(fingerprint('SOURCE', 'arm64-apple-macos27.0')).toHaveLength(12);
  });

  it('changes when the helper source or the OS changes', () => {
    const base = fingerprint('a', 't');
    expect(fingerprint('b', 't')).not.toBe(base);
    expect(fingerprint('a', 'u')).not.toBe(base);
  });
});

describe('embedded helper source', () => {
  it('is byte-identical to swift/helper.swift', () => {
    // The two packages ship copies of one file; this is what stops them drifting.
    const source = readFileSync(path.join(repoRoot, 'swift', 'helper.swift'));
    const embedded = readFileSync(path.join(repoRoot, 'packages', 'node', 'swift', 'helper.swift'));
    expect(createHash('sha256').update(embedded).digest('hex')).toBe(
      createHash('sha256').update(source).digest('hex'),
    );
  });

  it('matches the Python package copy too', () => {
    const source = readFileSync(path.join(repoRoot, 'swift', 'helper.swift'));
    const python = readFileSync(
      path.join(repoRoot, 'packages', 'python', 'src', 'apple_llm', 'swift', 'helper.swift'),
    );
    expect(createHash('sha256').update(python).digest('hex')).toBe(
      createHash('sha256').update(source).digest('hex'),
    );
  });
});

describe('JSON recovery from free text', () => {
  // Only the cloud tier needs this: on device, constrained decoding means the
  // reply is already JSON.
  it('parses bare JSON', () => {
    expect(parseLlmJson('{"a":1}')).toEqual({ a: 1 });
  });

  it('strips a leading fence', () => {
    expect(parseLlmJson('```json\n{"a":1}\n```')).toEqual({ a: 1 });
  });

  it('strips an untagged fence', () => {
    expect(parseLlmJson('```\n{"a":1}\n```')).toEqual({ a: 1 });
  });

  it('finds a fence after prose', () => {
    expect(parseLlmJson('Here is the JSON:\n```json\n{"a":1}\n```\nHope that helps!')).toEqual({ a: 1 });
  });

  it('digs an object out of surrounding prose with no fence', () => {
    expect(parseLlmJson('Sure! {"a":1} — let me know.')).toEqual({ a: 1 });
  });

  it('recovers a top-level array', () => {
    expect(parseLlmJson('Result: [1,2,3]')).toEqual([1, 2, 3]);
  });

  it('is not fooled by braces inside strings', () => {
    expect(parseLlmJson('x {"a":"}{ not the end"} y')).toEqual({ a: '}{ not the end' });
  });

  it('is not fooled by an escaped quote inside a string', () => {
    expect(parseLlmJson('{"a":"say \\"hi\\" }"}')).toEqual({ a: 'say "hi" }' });
  });

  it('handles nested objects', () => {
    expect(parseLlmJson('note {"a":{"b":[{"c":1}]}} end')).toEqual({ a: { b: [{ c: 1 }] } });
  });

  it('throws with an echo of the reply when there is no JSON', () => {
    expect(() => parseLlmJson('I cannot help with that.')).toThrow(/I cannot help with that/);
  });

  it('exposes the pieces', () => {
    expect(stripCodeFences('```json\n{"a":1}\n```')).toBe('{"a":1}');
    expect(extractJsonSpan('nope')).toBeNull();
    expect(extractJsonSpan('a {"b":1} c')).toBe('{"b":1}');
  });
});

describe('image attachments', () => {
  it('passes a bare path through', () => {
    expect(parseImageFlag('./photo.png')).toBe('./photo.png');
  });

  it('splits path::label', () => {
    expect(parseImageFlag('./scan.png::receipt')).toEqual({ path: './scan.png', label: 'receipt' });
  });

  it('keeps Windows-style drive letters intact (no label split on single colon)', () => {
    expect(parseImageFlag('C:/photos/a.png')).toBe('C:/photos/a.png');
  });

  it('ignores an empty label', () => {
    expect(parseImageFlag('./a.png::')).toBe('./a.png::');
  });
});

describe('built-in tools', () => {
  it('accepts the three known tools', () => {
    expect(() => assertTools(['ocr', 'barcode', 'spotlight'])).not.toThrow();
    expect(() => assertTools(undefined)).not.toThrow();
  });

  it('rejects unknown tools before spending a model call', () => {
    expect(() => assertTools(['laser' as never])).toThrow(/unknown tool/i);
  });
});

describe('document inlining', () => {
  it('returns the prompt unchanged with no documents', async () => {
    expect(await withDocuments('hi', undefined)).toBe('hi');
    expect(await withDocuments('hi', [])).toBe('hi');
  });

  it('splices text files with a delimiter', async () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'apple-llm-test-'));
    const file = path.join(dir, 'note.txt');
    writeFileSync(file, 'remember the milk', 'utf8');
    const out = await withDocuments('Summarize:', [file]);
    expect(out).toContain('Summarize:');
    expect(out).toContain('--- Document:');
    expect(out).toContain('remember the milk');
  });

  it('refuses a missing file rather than silently dropping it', async () => {
    await expect(withDocuments('hi', ['/definitely/not/here.txt'])).rejects.toThrow(/not found/i);
  });
});
