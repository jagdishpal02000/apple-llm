/**
 * Schema interop without a dependency.
 *
 * Any library implementing Standard Schema (https://standardschema.dev) *and*
 * Standard JSON Schema — Zod 4.2+, ArkType 2.1+, Valibot through
 * `toStandardJsonSchema()` — can be passed wherever a JSON Schema is accepted.
 * The JSON Schema goes to Apple's decoder; the library's own `validate` runs on
 * the reply, so refinements the decoder cannot enforce (`.email()`, `.min(3)`,
 * transforms, defaults) still apply, and the result is typed by the library.
 *
 * The interfaces below are copied from the spec rather than imported: the spec
 * is designed to be vendored, and the package stays at zero dependencies.
 */
import { SchemaValidationError, UnsupportedError, type Tier } from './errors.js';
import type { JsonSchema } from './schema.js';

export interface StandardSchemaV1<Input = unknown, Output = Input> {
  readonly '~standard': StandardSchemaV1.Props<Input, Output>;
}

export declare namespace StandardSchemaV1 {
  interface Props<Input = unknown, Output = Input> {
    readonly version: 1;
    readonly vendor: string;
    readonly validate: (value: unknown) => Result<Output> | Promise<Result<Output>>;
    readonly types?: Types<Input, Output> | undefined;
  }
  type Result<Output> = SuccessResult<Output> | FailureResult;
  interface SuccessResult<Output> {
    readonly value: Output;
    readonly issues?: undefined;
  }
  interface FailureResult {
    readonly issues: ReadonlyArray<Issue>;
  }
  interface Issue {
    readonly message: string;
    readonly path?: ReadonlyArray<PropertyKey | { readonly key: PropertyKey }> | undefined;
  }
  interface Types<Input = unknown, Output = Input> {
    readonly input: Input;
    readonly output: Output;
  }
}

/** The Standard JSON Schema extension: a schema that can describe itself. */
export interface StandardJSONSchemaV1 {
  readonly '~standard': {
    readonly jsonSchema: {
      readonly input: (options: { target: string }) => Record<string, unknown>;
      readonly output: (options: { target: string }) => Record<string, unknown>;
    };
  };
}

/** Anything `json()`, `streamJson()` and `tool()` accept as a schema. */
export type SchemaLike = JsonSchema | StandardSchemaV1;

/** The parsed type a schema produces: inferred for Standard Schemas, `unknown` for plain JSON Schema. */
export type InferSchema<S> = S extends StandardSchemaV1<any, infer Output> ? Output : unknown;

export function isStandardSchema(value: unknown): value is StandardSchemaV1 {
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) return false;
  const props = (value as Record<string, unknown>)['~standard'];
  return props !== null && typeof props === 'object' && typeof (props as { validate?: unknown }).validate === 'function';
}

/** A schema reduced to the two things this package needs from it. */
export interface ResolvedSchema {
  /** Plain JSON Schema, before the Apple-dialect rewrite. */
  json: JsonSchema;
  /** The library's validator, when there is one. */
  validate?: (value: unknown) => Promise<StandardSchemaV1.Result<unknown>>;
}

export function resolveSchema(schema: SchemaLike): ResolvedSchema {
  if (!isStandardSchema(schema)) return { json: schema as JsonSchema };
  const props = schema['~standard'];
  const describe = (props as unknown as StandardJSONSchemaV1['~standard']).jsonSchema;
  if (describe === undefined || typeof describe.input !== 'function') {
    throw new UnsupportedError(
      `This ${props.vendor} schema cannot describe itself as JSON Schema, which Apple's decoder needs.\n` +
        (props.vendor === 'valibot'
          ? 'Wrap it: `toStandardJsonSchema(schema)` from @valibot/to-json-schema.'
          : 'Use a version that implements Standard JSON Schema (zod >= 4.2, arktype >= 2.1), or pass a JSON Schema.'),
    );
  }
  // The *input* schema: the model produces what the validator consumes, so a
  // transform or default applies to the model's output rather than being
  // demanded of it.
  let json: Record<string, unknown>;
  try {
    json = describe.input({ target: 'draft-2020-12' });
  } catch {
    json = describe.input({ target: 'draft-07' });
  }
  return { json, validate: async (value) => props.validate(value) };
}

function formatPath(path: StandardSchemaV1.Issue['path']): string {
  if (path === undefined || path.length === 0) return '';
  return path
    .map((segment) => (typeof segment === 'object' && segment !== null ? segment.key : segment))
    .map((key) => String(key))
    .join('.');
}

export function describeIssues(issues: ReadonlyArray<StandardSchemaV1.Issue>): string {
  return issues
    .slice(0, 8)
    .map((issue) => {
      const at = formatPath(issue.path);
      return at === '' ? issue.message : `${at}: ${issue.message}`;
    })
    .join('; ');
}

/** Run a resolved schema's validator, throwing a typed error on failure. */
export async function validateWith(
  resolved: ResolvedSchema,
  value: unknown,
  text: string,
  tier: Tier,
): Promise<unknown> {
  if (resolved.validate === undefined) {
    // A plain JSON Schema has no validator of its own, but its patterns were
    // stripped for Apple's decoder and must still hold.
    const issues = checkPatterns(value, resolved.json);
    if (issues.length === 0) return value;
    throw new SchemaValidationError(
      `The model's reply did not satisfy the schema: ${describeIssues(issues)}`,
      issues.map((issue) => ({ message: issue.message, path: issue.path })),
      text,
      tier,
    );
  }
  const result = await resolved.validate(value);
  if (result.issues !== undefined) {
    throw new SchemaValidationError(
      `The model's reply did not satisfy the schema: ${describeIssues(result.issues)}`,
      result.issues.map((issue) => ({ message: issue.message, path: issue.path })),
      text,
      tier,
    );
  }
  return result.value;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Does this (sub)schema admit `null`? */
function allowsNull(schema: unknown): boolean {
  if (!isObject(schema)) return false;
  if (schema.type === 'null' || schema.nullable === true) return true;
  if (Array.isArray(schema.type) && schema.type.includes('null')) return true;
  if (Array.isArray(schema.enum) && schema.enum.includes(null)) return true;
  for (const key of ['anyOf', 'oneOf']) {
    const branches = schema[key];
    if (Array.isArray(branches) && branches.some(allowsNull)) return true;
  }
  return false;
}

/**
 * Put back the `null`s Apple's dialect took away.
 *
 * Apple has no union types, so a property that is required-but-nullable is
 * rewritten as simply not required, and the model then omits it rather than
 * writing `null`. Your schema still says the key must be present, so a
 * validator (or plain code reading `obj.key === null`) would reject or
 * misread the reply. This walks the *original* schema and fills each such
 * missing key with `null`. Never overwrites a value the model produced.
 */
export function restoreNulls(value: unknown, schema: JsonSchema, root: JsonSchema = schema, depth = 0): unknown {
  if (depth > 32 || !isObject(schema)) return value;
  const ref = schema.$ref;
  if (typeof ref === 'string') {
    const target = resolveRef(ref, root);
    return target === undefined ? value : restoreNulls(value, target, root, depth + 1);
  }
  for (const key of ['anyOf', 'oneOf']) {
    const branches = schema[key];
    if (Array.isArray(branches)) {
      // Only the unambiguous case: one non-null branch (pydantic's Optional[Model]).
      const nonNull = branches.filter((b) => !(isObject(b) && b.type === 'null'));
      if (nonNull.length === 1 && isObject(nonNull[0])) {
        return restoreNulls(value, nonNull[0] as JsonSchema, root, depth + 1);
      }
      return value;
    }
  }
  if (Array.isArray(schema.allOf) && schema.allOf.length === 1 && isObject(schema.allOf[0])) {
    return restoreNulls(value, schema.allOf[0] as JsonSchema, root, depth + 1);
  }
  if (Array.isArray(value)) {
    return isObject(schema.items) ? value.map((item) => restoreNulls(item, schema.items as JsonSchema, root, depth + 1)) : value;
  }
  if (!isObject(value) || !isObject(schema.properties)) return value;
  const properties = schema.properties;
  const required = Array.isArray(schema.required) ? (schema.required as unknown[]) : [];
  const out: Record<string, unknown> = { ...value };
  for (const [key, child] of Object.entries(properties)) {
    if (out[key] === undefined) {
      if (required.includes(key) && allowsNull(child)) out[key] = null;
      continue;
    }
    if (isObject(child)) out[key] = restoreNulls(out[key], child as JsonSchema, root, depth + 1);
  }
  return out;
}

function resolveRef(ref: string, root: JsonSchema): JsonSchema | undefined {
  const match = /^#\/(\$defs|definitions)\/(.+)$/.exec(ref);
  if (match === null) return ref === '#' ? root : undefined;
  const defs = root[match[1]];
  const target = isObject(defs) ? defs[decodeURIComponent(match[2])] : undefined;
  return isObject(target) ? (target as JsonSchema) : undefined;
}

/**
 * Check the `pattern`s Apple's decoder cannot enforce (dialect rule 9) against
 * a reply, for plain JSON Schemas — a Standard Schema's own validator already
 * covers them. Returns the issues; empty when every pattern holds. A pattern
 * JavaScript cannot compile is skipped rather than failing the reply.
 */
export function checkPatterns(
  value: unknown,
  schema: JsonSchema,
  root: JsonSchema = schema,
  path: PropertyKey[] = [],
  depth = 0,
): StandardSchemaV1.Issue[] {
  if (depth > 32 || !isObject(schema)) return [];
  if (typeof schema.$ref === 'string') {
    const target = resolveRef(schema.$ref, root);
    return target === undefined ? [] : checkPatterns(value, target, root, path, depth + 1);
  }
  if (Array.isArray(schema.allOf) && schema.allOf.length === 1 && isObject(schema.allOf[0])) {
    return checkPatterns(value, schema.allOf[0] as JsonSchema, root, path, depth + 1);
  }
  for (const key of ['anyOf', 'oneOf']) {
    const branches = schema[key];
    if (!Array.isArray(branches)) continue;
    const nonNull = branches.filter((b) => !(isObject(b) && b.type === 'null'));
    // Only the unambiguous case; with several branches the value may match any.
    return nonNull.length === 1 && isObject(nonNull[0]) && value !== null
      ? checkPatterns(value, nonNull[0] as JsonSchema, root, path, depth + 1)
      : [];
  }
  if (typeof value === 'string' && typeof schema.pattern === 'string') {
    let regex: RegExp | undefined;
    try {
      regex = new RegExp(schema.pattern, 'u');
    } catch {
      regex = undefined;
    }
    return regex === undefined || regex.test(value)
      ? []
      : [{ message: `must match the pattern ${schema.pattern}`, path }];
  }
  if (Array.isArray(value) && isObject(schema.items)) {
    return value.flatMap((item, i) => checkPatterns(item, schema.items as JsonSchema, root, [...path, i], depth + 1));
  }
  if (isObject(value) && isObject(schema.properties)) {
    return Object.entries(schema.properties).flatMap(([key, child]) =>
      value[key] === undefined || !isObject(child)
        ? []
        : checkPatterns(value[key], child as JsonSchema, root, [...path, key], depth + 1),
    );
  }
  return [];
}
