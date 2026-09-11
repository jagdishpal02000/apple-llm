/**
 * Apple's `GenerationSchema` is `Decodable` from JSON Schema, but only accepts a
 * restricted dialect. These rules were established empirically in api-scribe, by
 * decoding a real schema against macOS 26 and 27 until it was accepted:
 *
 *  1. Union types are rejected — `type: ['string','null']` fails with
 *     "Expected value of type String". Nullability is expressed by leaving the
 *     property out of `required` instead.
 *  2. Every object schema requires an `x-order` array naming its properties in
 *     generation order. Omitting it fails with "Key 'x-order' not found".
 *  3. `enum` is not recognized. A node may only carry `type`, `const`, `$ref` or
 *     `anyOf`, so enums become `anyOf` of `const` branches.
 *  4. Every object schema *and* every `anyOf` schema requires a `title`, and
 *     `$ref` resolves by that title rather than by JSON pointer — so a
 *     definition's title must equal the key it is referenced by.
 *  5. Every object schema must state `additionalProperties`; omitting it fails
 *     with "Key 'additionalProperties' not found". macOS 26's decoder tolerated
 *     its absence, so this one only surfaced on macOS 27.
 *
 *  6. `const` is decoded as a String, always. A numeric member (`const: 200`)
 *     fails with "Expected value of type String". Stringifying it is accepted
 *     but changes the output type — the model then emits `"200"` rather than
 *     `200` — so a numeric enum is converted to its underlying type instead and
 *     the literal constraint is dropped. api-scribe reached the same conclusion
 *     the hard way: it settled on a plain `integer` for HTTP status after a
 *     string-typed union produced junk like `": 201"`.
 *  7. An object with no properties still needs an explicit `properties: {}`.
 *     Without it Apple reads `additionalProperties` in its other JSON Schema
 *     sense — a schema for the values — and fails with "Expected value of type
 *     Dictionary<String, Any>". Supplying a dictionary there is accepted but
 *     turns the field into a free-form map the model fills with invented keys.
 *
 *  8. A `title` on a `type: "string"` node is rejected: Apple reads it as a
 *     "named string type" and fails with "Named string types must have a
 *     non-empty enum field". Other primitives (integer, number, boolean) and
 *     arrays accept a title happily. Titles are therefore stripped from plain
 *     strings. This matters most for pydantic, whose `model_json_schema()`
 *     titles every single property.
 *
 * Rules 6, 7 and 8 were found by decoding this package's own fixture corpus
 * against macOS 27. api-scribe hit none of them: it had only string enums, no
 * empty objects, and hand-written schemas that never titled a string.
 *
 * `$ref` / `$defs` are otherwise supported and pass through untouched.
 *
 * Why this matters more than it looks: constrained decoding makes a schema
 * mistake invisible but total. Collapsing a `["number","string"]` union to
 * `"string"` made it physically impossible for the model to emit a status code —
 * it wrote `": 201"` and the literal `"default"` instead. Never collapse a
 * multi-type union to one branch; convert it to `anyOf`.
 */

export type JsonSchema = Record<string, unknown>;

/** Keys whose value is a map of name -> schema. */
const SCHEMA_MAP_KEYS = ['properties', '$defs', 'definitions'];
/** Keys whose value is an array of schemas. */
const SCHEMA_LIST_KEYS = ['anyOf', 'oneOf', 'allOf'];
/** Keys whose value is a single schema. */
const SCHEMA_KEYS = ['items', 'not', 'additionalItems'];

function isPlainObject(value: unknown): value is JsonSchema {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Apple uses `title` as the generated type name *and* resolves `$ref` against it
 * rather than by JSON pointer, so a definition's title has to match the key it
 * is referenced by — `#/$defs/param` only resolves to a schema titled `param`.
 * That rules out any prettifying of the name here.
 */
function sanitize(name: string, counter: { n: number }): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/g, '');
  if (cleaned.length === 0 || /^[0-9]/.test(cleaned)) {
    counter.n += 1;
    return `Schema${counter.n}`;
  }
  return cleaned;
}

/**
 * A title that is not already taken.
 *
 * Titles are Apple's generated type names, so two structurally different objects
 * sharing one is a real collision and not just untidy — nested objects take
 * their title from the property that holds them, and a schema with `items` at
 * two different levels would otherwise produce two different types both called
 * `items`. Suffixing keeps them distinct. Definition titles are reserved up
 * front and never reassigned here, since a `$ref` has to keep matching.
 */
function titleFrom(name: string, ctx: Context): string {
  const base = sanitize(name, ctx.counter);
  if (!ctx.used.has(base)) {
    ctx.used.add(base);
    return base;
  }
  for (let i = 2; ; i += 1) {
    const candidate = `${base}${i}`;
    if (!ctx.used.has(candidate)) {
      ctx.used.add(candidate);
      return candidate;
    }
  }
}

/** JSON Schema type name for an enum member. */
function memberType(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
  if (typeof value === 'boolean') return 'boolean';
  return 'string';
}

interface ConvertedNode {
  schema: JsonSchema;
  /** True when the source node allowed `null`, which Apple models as "not required". */
  nullable: boolean;
}

interface Context {
  counter: { n: number };
  /**
   * Definition key -> the title that definition will carry. Computed in one
   * pass up front so a `$ref` can be rewritten to match: sanitizing `my-def`
   * to `mydef` would otherwise leave `#/$defs/my-def` resolving to nothing,
   * silently, since Apple resolves by title. Only the ref string is rewritten
   * and not the `$defs` key; if a future macOS ever resolves by pointer
   * instead, the two would have to move together.
   */
  defTitles: Map<string, string>;
  /** Every title handed out so far, so generated ones stay unique. */
  used: Set<string>;
}

function rewriteRef(ref: string, ctx: Context): string {
  const slash = ref.lastIndexOf('/');
  if (slash === -1) return ref;
  const name = ref.slice(slash + 1);
  const title = ctx.defTitles.get(name);
  return title === undefined || title === name ? ref : `${ref.slice(0, slash + 1)}${title}`;
}

function convertNode(node: JsonSchema, name: string, ctx: Context): ConvertedNode {
  const out: JsonSchema = {};
  let nullable = false;

  // Convert child properties first: a nullable child decides the parent's `required`.
  const converted = new Map<string, ConvertedNode>();
  const properties = node.properties;
  if (isPlainObject(properties)) {
    for (const [key, value] of Object.entries(properties)) {
      converted.set(
        key,
        isPlainObject(value)
          ? convertNode(value, key, ctx)
          : { schema: value as JsonSchema, nullable: false },
      );
    }
  }

  for (const [key, value] of Object.entries(node)) {
    if (key === 'type' || key === 'enum' || key === 'required' || key === 'properties') continue;

    if (key === '$ref' && typeof value === 'string') {
      out.$ref = rewriteRef(value, ctx);
    } else if (SCHEMA_MAP_KEYS.includes(key) && isPlainObject(value)) {
      const isDefs = key === '$defs' || key === 'definitions';
      const mapped: JsonSchema = {};
      for (const [childName, childValue] of Object.entries(value)) {
        if (!isPlainObject(childValue)) {
          mapped[childName] = childValue;
          continue;
        }
        const child = convertNode(childValue, childName, ctx).schema;
        // Definitions take the title reserved for them in the pre-pass, so the
        // rewritten `$ref`s above point at a title that actually exists.
        const reserved = isDefs ? ctx.defTitles.get(childName) : undefined;
        if (reserved !== undefined) child.title = reserved;
        mapped[childName] = child;
      }
      out[key] = mapped;
    } else if (SCHEMA_LIST_KEYS.includes(key) && Array.isArray(value)) {
      // Apple only understands `anyOf`; `oneOf` is the same shape for our purposes.
      // `allOf` is different (must match all, not one) and Apple rejects the key
      // outright ("None of these keys were present: 'type', 'const', '$ref',
      // 'anyOf'"). A single-element `allOf` — the common pydantic pattern
      // `{description, allOf: [{$ref}]}` — unwraps to that entry; multi-element
      // `allOf` is left to fail loudly as SchemaRejectedError rather than
      // silently mistranslated to `anyOf`.
      if (key === 'allOf') {
        const allOfConverted = value.map((entry) =>
          isPlainObject(entry) ? convertNode(entry, name, ctx).schema : entry,
        );
        if (allOfConverted.length === 1 && isPlainObject(allOfConverted[0])) {
          for (const [k, v] of Object.entries(allOfConverted[0] as JsonSchema)) {
            if (out[k] === undefined) out[k] = v;
          }
        } else {
          out.allOf = allOfConverted;
        }
      } else {
        const target = key === 'oneOf' ? 'anyOf' : key;
        out[target] = value.map((entry) =>
          isPlainObject(entry) ? convertNode(entry, name, ctx).schema : entry,
        );
      }
    } else if (SCHEMA_KEYS.includes(key) && isPlainObject(value)) {
      out[key] = convertNode(value, name, ctx).schema;
    } else {
      out[key] = value;
    }
  }

  // Rule 1: split union types. `null` becomes optionality; a genuine multi-type
  // union becomes `anyOf`, which Apple does understand. Collapsing such a union
  // to a single type would make the other branches unreachable under constrained
  // decoding — the model then cannot emit them at all, however hard it tries.
  const rawType = node.type;
  if (Array.isArray(rawType)) {
    const nonNull = rawType.filter((t) => t !== 'null');
    nullable = nonNull.length !== rawType.length;
    if (nonNull.length > 1) {
      out.anyOf = nonNull.map((t) => ({ type: t }));
    } else {
      out.type = nonNull[0] ?? 'string';
    }
  } else if (rawType !== undefined) {
    out.type = rawType;
  }

  // Rule 3: enums become anyOf/const branches. A `null` member is dropped and
  // recorded as nullability, matching rule 1.
  const rawEnum = node.enum;
  if (Array.isArray(rawEnum) && rawEnum.length > 0) {
    const members = rawEnum.filter((v) => v !== null);
    if (members.length !== rawEnum.length) nullable = true;
    if (members.length > 0) {
      delete out.type;
      if (members.every((v) => typeof v === 'string')) {
        out.anyOf = members.map((value) => ({ type: 'string', const: value }));
      } else {
        // Rule 6: `const` must be a String, so a non-string enum cannot be
        // expressed as const branches. Keep the type and drop the literal
        // constraint rather than stringify — a silently retyped field is the
        // worse failure, and under constrained decoding the caller cannot see
        // it happen. Mention the allowed values in the field description if
        // they matter.
        const types = [...new Set(members.map(memberType))];
        if (types.length === 1) out.type = types[0];
        else out.anyOf = types.map((t) => ({ type: t }));
      }
    }
  }

  const isObject = out.type === 'object' || converted.size > 0;
  if (isObject) {
    const props: JsonSchema = {};
    for (const [key, child] of converted) props[key] = child.schema;
    // Rule 7: `properties` must be present even when empty, or Apple reads
    // `additionalProperties` as a value schema and rejects the boolean.
    out.properties = props;

    const originalRequired = Array.isArray(node.required) ? (node.required as string[]) : [];
    // Rule 1 (cont.): a nullable property is modelled as simply not required.
    out.required = originalRequired.filter((key) => !converted.get(key)?.nullable);
    // Rule 2: object property order must be declared. A caller-supplied x-order
    // is honoured — it is the generation order, so a schema whose later fields
    // depend on earlier ones has a real reason to choose it — but any property
    // missing from it is appended, since Apple requires every one to be listed.
    const keys = Object.keys(props);
    const declaredOrder = Array.isArray(node['x-order'])
      ? (node['x-order'] as unknown[]).filter((k): k is string => typeof k === 'string' && keys.includes(k))
      : [];
    out['x-order'] = [...declaredOrder, ...keys.filter((k) => !declaredOrder.includes(k))];
    // Rule 4: objects need a title.
    if (typeof out.title === 'string') ctx.used.add(out.title);
    else out.title = titleFrom(name, ctx);
    // Rule 5: every object schema must state `additionalProperties`.
    if (out.additionalProperties === undefined) out.additionalProperties = false;
  } else if (Array.isArray(node.required)) {
    out.required = node.required;
  }

  // Rule 4 (cont.): anyOf/allOf schemas need a title too. Apple only
  // understands `anyOf`, but `allOf` passes through with the same title
  // requirement (different composition semantics, so the key is kept).
  if (
    (Array.isArray(out.anyOf) || Array.isArray((out as any).allOf)) &&
    typeof out.title !== 'string'
  ) {
    out.title = titleFrom(name, ctx);
  }

  // Rule 8: a titled string is a "named string type" and must carry an enum.
  // Strip the title instead — it is only a generated type name, and a plain
  // string does not need one. (A `$defs` entry that is a bare string would lose
  // its `$ref` target here, but such a schema cannot be expressed on Apple at
  // all: a named string type has to be an enum.)
  if (out.type === 'string' && out.anyOf === undefined && (out as any).allOf === undefined)
    delete out.title;

  return { schema: out, nullable };
}

/**
 * Reserve each definition's title before converting anything, using the same
 * counter the conversion will use. Doing it in a throwaway counter instead
 * would let a generated `SchemaN` name drift between the two passes.
 */
function reserveDefTitles(schema: JsonSchema, ctx: Context): void {
  for (const key of ['$defs', 'definitions']) {
    const defs = schema[key];
    if (!isPlainObject(defs)) continue;
    for (const [name, value] of Object.entries(defs)) {
      const declared = isPlainObject(value) && typeof value.title === 'string' ? value.title : null;
      const title = declared ?? sanitize(name, ctx.counter);
      ctx.defTitles.set(name, title);
      ctx.used.add(title);
    }
  }
}

/**
 * Translate a standard JSON Schema into the dialect Apple's `GenerationSchema`
 * accepts. Constrained decoding then makes the shape of the reply a guarantee
 * rather than a request.
 */
export function toAppleSchema(schema: JsonSchema, rootName = 'Response'): JsonSchema {
  const ctx: Context = { counter: { n: 0 }, defTitles: new Map(), used: new Set() };
  reserveDefTitles(schema, ctx);
  return convertNode(schema, rootName, ctx).schema;
}
