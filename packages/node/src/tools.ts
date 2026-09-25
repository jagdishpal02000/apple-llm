/**
 * Function tools: tools you define in JavaScript, which the on-device model
 * can call mid-generation.
 *
 * The model decides to call one; the helper sends the call up as an event; the
 * tool's `execute` runs here, in your process, with your credentials and your
 * database; its result goes back down and generation continues. Arguments are
 * produced under constrained decoding against the tool's parameter schema, so
 * they always match it — the same guarantee `json()` gives.
 *
 * A tool without `execute` is *client-driven*: generation stops at the call
 * and the result reports it with `finishReason: 'tool-calls'`, for a caller
 * that runs tools itself (the OpenAI-compatible server and the AI SDK provider
 * work this way). Send the results back as `tool` messages to continue.
 */
import { AppleLLMError } from './errors.js';
import type { JsonSchema } from './schema.js';
import {
  resolveSchema,
  type InferSchema,
  type ResolvedSchema,
  type SchemaLike,
  type StandardSchemaV1,
} from './standard-schema.js';

/** Apple's built-in on-device tools (all macOS 27+, all local). */
export type BuiltInTool = 'ocr' | 'barcode' | 'spotlight';

export const BUILT_IN_TOOLS: ReadonlySet<string> = new Set(['ocr', 'barcode', 'spotlight']);

export interface ToolExecutionContext {
  /** The id of this call, as it will appear in `toolCalls`. */
  toolCallId: string;
  /** Fires when the request is aborted; long-running tools should honour it. */
  signal?: AbortSignal;
}

export interface FunctionTool<Args = any, Result = unknown> {
  /** What the model calls it. Letters, digits, `_` and `-`; must start with a letter or `_`. */
  name: string;
  /** When to use it. The model reads this, so write it for the model. */
  description?: string;
  /** JSON Schema or a Standard Schema (Zod, ArkType, …). Omit for a tool that takes no arguments. */
  parameters?: SchemaLike;
  /**
   * Runs the tool. Return a string, or anything JSON-serialisable. Omit it for
   * a client-driven tool that stops generation instead (see above).
   * A thrown error ends the request with a ToolExecutionError whose `cause` is
   * your error; return an error *message* instead if the model should see it
   * and recover.
   */
  execute?: (args: Args, context: ToolExecutionContext) => Result | Promise<Result>;
}

/**
 * Define a function tool, with the argument type inferred from a Standard
 * Schema.
 *
 *   const weather = tool({
 *     name: 'getWeather',
 *     description: 'Current weather for a city',
 *     parameters: z.object({ city: z.string() }),
 *     execute: async ({ city }) => fetchWeather(city),   // city: string
 *   });
 */
export function tool<S extends SchemaLike, Result = unknown>(definition: {
  name: string;
  description?: string;
  parameters?: S;
  execute?: (args: InferSchema<S>, context: ToolExecutionContext) => Result | Promise<Result>;
}): FunctionTool<InferSchema<S>, Result> {
  return definition as FunctionTool<InferSchema<S>, Result>;
}

/** Everything `tools` accepts: built-in names, function tools, or a record of function tools keyed by name. */
export type ToolSet =
  | ReadonlyArray<BuiltInTool | FunctionTool>
  | Readonly<Record<string, Omit<FunctionTool, 'name'> & { name?: string }>>;

export interface NormalizedTools {
  builtIn: BuiltInTool[];
  functions: Array<FunctionTool & { resolved?: ResolvedSchema; json: JsonSchema }>;
}

const TOOL_NAME = /^[A-Za-z_][A-Za-z0-9_-]{0,63}$/;

const EMPTY_PARAMETERS: JsonSchema = { type: 'object', properties: {}, required: [] };

/** Split and check a tool set. Throws before any model call on a bad definition. */
export function normalizeTools(tools: ToolSet | undefined): NormalizedTools {
  const out: NormalizedTools = { builtIn: [], functions: [] };
  if (tools === undefined) return out;
  const entries: Array<BuiltInTool | FunctionTool> = Array.isArray(tools)
    ? [...(tools as ReadonlyArray<BuiltInTool | FunctionTool>)]
    : Object.entries(tools as Record<string, FunctionTool>).map(([name, def]) => ({ ...def, name: def.name ?? name }));
  const seen = new Set<string>();
  for (const entry of entries) {
    if (typeof entry === 'string') {
      if (!BUILT_IN_TOOLS.has(entry)) {
        throw new AppleLLMError(`Unknown tool "${entry}" (want ${[...BUILT_IN_TOOLS].join(', ')}).`, 'device');
      }
      if (!out.builtIn.includes(entry)) out.builtIn.push(entry);
      continue;
    }
    if (entry === null || typeof entry !== 'object' || typeof entry.name !== 'string') {
      throw new AppleLLMError('A function tool needs a name.', 'device');
    }
    if (!TOOL_NAME.test(entry.name)) {
      throw new AppleLLMError(
        `Tool name "${entry.name}" is invalid: use letters, digits, "_" or "-", starting with a letter or "_".`,
        'device',
      );
    }
    if (seen.has(entry.name) || BUILT_IN_TOOLS.has(entry.name)) {
      throw new AppleLLMError(`Tool name "${entry.name}" is used twice.`, 'device');
    }
    seen.add(entry.name);
    const resolved = entry.parameters === undefined ? undefined : resolveSchema(entry.parameters);
    out.functions.push({ ...entry, resolved, json: resolved?.json ?? EMPTY_PARAMETERS });
  }
  return out;
}

/** Tool output as the model will read it. */
export function toolOutputText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** Re-exported so `tool()` users can type things without reaching into internals. */
export type { StandardSchemaV1, SchemaLike, InferSchema };
