import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { toAppleSchema } from '../src/schema.js';

const here = path.dirname(fileURLToPath(import.meta.url));
/**
 * The corpus is shared with the Python package on purpose: both suites read
 * these same files, which is what keeps the two ports honest.
 */
const FIXTURES = path.resolve(here, '..', '..', '..', 'tests', 'fixtures', 'schema');

interface Fixture {
  name: string;
  why: string;
  input: Record<string, unknown>;
  expected: Record<string, unknown>;
}

const fixtures: Fixture[] = readdirSync(FIXTURES)
  .filter((f) => f.endsWith('.json'))
  .sort()
  .map((f) => JSON.parse(readFileSync(path.join(FIXTURES, f), 'utf8')) as Fixture);

describe('toAppleSchema golden corpus', () => {
  it('has fixtures', () => {
    expect(fixtures.length).toBeGreaterThan(10);
  });

  for (const fixture of fixtures) {
    it(`${fixture.name}: ${fixture.why}`, () => {
      expect(toAppleSchema(fixture.input)).toEqual(fixture.expected);
    });
  }
});

describe('the dialect rules, stated directly', () => {
  it('rule 1: never collapses a multi-type union to one branch', () => {
    // Collapsing ["number","string"] to "string" made it physically impossible
    // for the model to emit a status code; it wrote ": 201" instead.
    const out = toAppleSchema({ type: 'object', properties: { s: { type: ['number', 'string'] } } });
    const s = (out.properties as Record<string, any>).s;
    expect(s.anyOf).toEqual([{ type: 'number' }, { type: 'string' }]);
    expect(s.type).toBeUndefined();
  });

  it('rule 1: nullability is expressed by dropping the key from required', () => {
    const out = toAppleSchema({
      type: 'object',
      required: ['a', 'b'],
      properties: { a: { type: 'string' }, b: { type: ['string', 'null'] } },
    });
    expect((out.properties as any).b.type).toBe('string');
    expect(out.required).toEqual(['a']);
    // ...but it is still generated, so it must stay in x-order.
    expect(out['x-order']).toEqual(['a', 'b']);
  });

  it('rule 2: every object declares x-order covering every property', () => {
    const out = toAppleSchema({
      type: 'object',
      properties: { a: { type: 'string' }, nested: { type: 'object', properties: { z: { type: 'string' } } } },
    });
    expect(out['x-order']).toEqual(['a', 'nested']);
    expect((out.properties as any).nested['x-order']).toEqual(['z']);
  });

  it('rule 3: no bare enum survives', () => {
    const out = toAppleSchema({ type: 'object', properties: { e: { enum: ['x', 'y'] } } });
    const e = (out.properties as any).e;
    expect(e.enum).toBeUndefined();
    expect(e.anyOf).toEqual([
      { type: 'string', const: 'x' },
      { type: 'string', const: 'y' },
    ]);
  });

  it('rule 4: objects and anyOf nodes both carry a title', () => {
    const out = toAppleSchema({ type: 'object', properties: { e: { enum: ['x'] } } });
    expect(typeof out.title).toBe('string');
    expect(typeof (out.properties as any).e.title).toBe('string');
  });

  it('rule 4: $ref resolves by title, so a sanitized def name rewrites the ref', () => {
    const out = toAppleSchema({
      type: 'object',
      properties: { i: { $ref: '#/$defs/my-param' } },
      $defs: { 'my-param': { type: 'object', properties: { n: { type: 'string' } } } },
    });
    expect((out.$defs as any)['my-param'].title).toBe('myparam');
    expect((out.properties as any).i.$ref).toBe('#/$defs/myparam');
  });

  it('rule 5: every object states additionalProperties', () => {
    const out = toAppleSchema({ type: 'object', properties: { a: { type: 'string' } } });
    expect(out.additionalProperties).toBe(false);
  });

  it('rule 6: a numeric enum keeps its type rather than becoming string consts', () => {
    // `const` decodes as String only, so 200 cannot be a const. Retyping the
    // field to string would be the silent failure; dropping the literal is not.
    const out = toAppleSchema({ type: 'object', properties: { s: { enum: [200, 404] } } });
    const s = (out.properties as any).s;
    expect(s.type).toBe('integer');
    expect(s.anyOf).toBeUndefined();
    expect(JSON.stringify(s)).not.toContain('"200"');
  });

  it('rule 7: an object with no properties still gets properties: {}', () => {
    const out = toAppleSchema({ type: 'object', properties: { m: { type: 'object' } } });
    expect((out.properties as any).m.properties).toEqual({});
    expect((out.properties as any).m.additionalProperties).toBe(false);
  });

  it('honours a caller-supplied x-order, appending anything missing', () => {
    const out = toAppleSchema({
      type: 'object',
      'x-order': ['b'],
      properties: { a: { type: 'string' }, b: { type: 'string' } },
    });
    expect(out['x-order']).toEqual(['b', 'a']);
  });

  it('gives structurally different objects distinct titles', () => {
    // Two `items` at different levels would otherwise both be typed `items`.
    const out = toAppleSchema({
      type: 'object',
      properties: {
        items: { type: 'object', properties: { a: { type: 'string' } } },
        other: { type: 'array', items: { type: 'object', properties: { b: { type: 'string' } } } },
      },
    });
    const titles = [
      (out.properties as any).items.title,
      (out.properties as any).other.items.title,
    ];
    expect(new Set(titles).size).toBe(2);
  });

  it('is idempotent on its own output', () => {
    const once = toAppleSchema({
      type: 'object',
      required: ['a'],
      properties: { a: { type: 'string' }, b: { enum: ['x', 'y'] } },
    });
    expect(toAppleSchema(once)).toEqual(once);
  });
});
