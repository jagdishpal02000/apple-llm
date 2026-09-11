#!/usr/bin/env node
/**
 * Copy `swift/helper.swift` into both packages, and verify the copies match.
 *
 * The Swift source is shipped as package data rather than as a string literal
 * in each language. It carries backticks and backslash escapes, so a literal
 * would need a hand-maintained escaping layer in two languages — and a single
 * mis-escape there changes the source hash, which is the key the compiled
 * binary is cached under. Copying bytes cannot get that wrong.
 *
 *   node scripts/embed-helper.mjs           # regenerate both copies
 *   node scripts/embed-helper.mjs --check   # verify only, non-zero if stale
 */
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE = path.join(root, 'swift', 'helper.swift');

/** Where each package expects its copy. Both are gitignored-adjacent build output. */
export const COPIES = [
  path.join(root, 'packages', 'node', 'swift', 'helper.swift'),
  path.join(root, 'packages', 'python', 'src', 'apple_llm', 'swift', 'helper.swift'),
];

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

const check = process.argv.includes('--check');
const source = readFileSync(SOURCE);
const want = sha256(source);
let stale = 0;

for (const copy of COPIES) {
  let got = null;
  try {
    got = sha256(readFileSync(copy));
  } catch {
    /* missing counts as stale */
  }
  const rel = path.relative(root, copy);
  if (got === want) {
    console.log(`  ok    ${rel}`);
    continue;
  }
  stale += 1;
  if (check) {
    console.error(`  STALE ${rel} (${got ?? 'missing'} != ${want})`);
    continue;
  }
  mkdirSync(path.dirname(copy), { recursive: true });
  copyFileSync(SOURCE, copy);
  console.log(`  wrote ${rel}`);
}

console.log(`\nswift/helper.swift  sha256 ${want}`);
if (check && stale > 0) {
  console.error(`\n${stale} copy/copies are stale. Run: node scripts/embed-helper.mjs`);
  process.exit(1);
}
