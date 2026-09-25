/**
 * An OpenAI-compatible HTTP server for Apple's models.
 *
 *   apple-llm serve                        # http://127.0.0.1:11436/v1
 *
 * Point any OpenAI client at it — the official SDKs, LangChain, LlamaIndex,
 * editor plugins, Open WebUI — with any API key, and it talks to the on-device
 * model. Implements `POST /v1/chat/completions` (streaming, tools,
 * `response_format` with JSON Schema, images), `GET /v1/models` and
 * `GET /health`.
 *
 * Safe defaults, because a local model server is reachable by more than you
 * might think: it binds to 127.0.0.1 only; CORS is off, so a web page you
 * happen to visit cannot use your model; an API key can be required; and the
 * default tier is the device, so nothing leaves the machine unless a client
 * asks for the `apple-private-cloud` model by name.
 */
import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { TempImages } from './attachments.js';
import { AppleLLM, probe, type AppleLLMOptions, type GenerateResult, type TextOptions } from './client.js';
import type { SamplingMode, ToolCallRecord } from './device.js';
import {
  AbortError,
  AppleLLMError,
  ContextLengthError,
  ModelBusyError,
  ModelUnavailableError,
  QuotaError,
  RefusalError,
  SchemaRejectedError,
  SchemaValidationError,
  SetupRequiredError,
  TimeoutError,
  UnsupportedError,
} from './errors.js';
import type { ChatMessage } from './messages.js';
import type { JsonSchema } from './schema.js';
import { tool, type FunctionTool } from './tools.js';

export const DEFAULT_PORT = 11436;

/** Model ids a client can ask for. Anything else gets the server's default tier. */
export const MODELS = {
  'apple-on-device': 'device',
  'apple-private-cloud': 'cloud',
} as const;

export interface ServerOptions extends Omit<AppleLLMOptions, 'tier'> {
  port?: number;
  /** Interface to bind. Default 127.0.0.1; use 0.0.0.0 only on a network you trust. */
  host?: string;
  /** Tier for requests that do not name an Apple model. Default `device`. */
  tier?: 'device' | 'cloud' | 'auto';
  /** Require `Authorization: Bearer <apiKey>`. */
  apiKey?: string;
  /** `Access-Control-Allow-Origin` value, for browser clients. Off by default. */
  cors?: string;
  /** One line per request. Default: none. */
  log?: (line: string) => void;
}

/** OpenAI's error envelope, with a status code chosen from the typed error. */
function errorResponse(error: unknown): { status: number; body: Record<string, unknown> } {
  const message = error instanceof Error ? error.message : String(error);
  const make = (status: number, type: string, code?: string) => ({
    status,
    body: { error: { message, type, code: code ?? null, param: null } },
  });
  if (error instanceof ContextLengthError) return make(400, 'invalid_request_error', 'context_length_exceeded');
  if (error instanceof SchemaRejectedError || error instanceof SchemaValidationError) {
    return make(400, 'invalid_request_error', 'invalid_schema');
  }
  if (error instanceof UnsupportedError) return make(400, 'invalid_request_error', 'unsupported');
  if (error instanceof RefusalError) return make(400, 'invalid_request_error', 'content_filter');
  if (error instanceof QuotaError) return make(429, 'rate_limit_error', 'quota_exceeded');
  if (error instanceof TimeoutError) return make(504, 'timeout_error');
  if (error instanceof AbortError) return make(499, 'request_cancelled');
  if (error instanceof ModelBusyError) return make(503, 'service_unavailable', 'model_busy');
  if (error instanceof ModelUnavailableError || error instanceof SetupRequiredError) {
    return make(503, 'service_unavailable', 'model_unavailable');
  }
  if (error instanceof HttpError) return make(error.status, 'invalid_request_error');
  if (error instanceof AppleLLMError) return make(500, 'server_error');
  return make(500, 'server_error');
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

type Part = { type: string; text?: string; image_url?: string | { url: string } };

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (content === null || content === undefined) return '';
  if (Array.isArray(content)) {
    return (content as Part[])
      .filter((p) => p.type === 'text' || p.type === 'input_text')
      .map((p) => p.text ?? '')
      .join('\n');
  }
  return String(content);
}

interface OpenAIMessage {
  role: string;
  content?: unknown;
  tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
  name?: string;
}

async function toMessages(raw: unknown, images: TempImages, signal: AbortSignal): Promise<ChatMessage[]> {
  if (!Array.isArray(raw) || raw.length === 0) throw new HttpError(400, '`messages` must be a non-empty array.');
  const out: ChatMessage[] = [];
  for (const message of raw as OpenAIMessage[]) {
    switch (message.role) {
      case 'system':
      case 'developer':
        out.push({ role: 'system', content: textOf(message.content) });
        break;
      case 'user': {
        const paths: string[] = [];
        if (Array.isArray(message.content)) {
          for (const part of message.content as Part[]) {
            if (part.type === 'image_url' && part.image_url !== undefined) {
              const url = typeof part.image_url === 'string' ? part.image_url : part.image_url.url;
              paths.push(await images.add(url, undefined, signal));
            } else if (part.type !== 'text' && part.type !== 'input_text') {
              throw new HttpError(400, `Content part "${part.type}" is not supported.`);
            }
          }
        }
        out.push({ role: 'user', content: textOf(message.content), ...(paths.length > 0 ? { images: paths } : {}) });
        break;
      }
      case 'assistant':
        out.push({
          role: 'assistant',
          content: textOf(message.content),
          ...(message.tool_calls !== undefined && message.tool_calls.length > 0
            ? {
                toolCalls: message.tool_calls.map((call) => ({
                  id: call.id,
                  name: call.function.name,
                  arguments: parseArguments(call.function.arguments),
                })),
              }
            : {}),
        });
        break;
      case 'tool':
        out.push({ role: 'tool', toolCallId: String(message.tool_call_id ?? ''), name: message.name, content: textOf(message.content) });
        break;
      default:
        throw new HttpError(400, `Message role "${message.role}" is not supported.`);
    }
  }
  return out;
}

function parseArguments(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

interface ChatRequest {
  model?: string;
  messages?: unknown;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  temperature?: number;
  top_p?: number;
  seed?: number;
  n?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  response_format?: { type: string; json_schema?: { name?: string; schema?: JsonSchema } };
  tools?: Array<{ type: string; function?: { name: string; description?: string; parameters?: JsonSchema } }>;
  tool_choice?: unknown;
}

function openAIToolCalls(calls: ToolCallRecord[]): Array<Record<string, unknown>> {
  return calls
    .filter((call) => call.output === undefined)
    .map((call, index) => ({
      index,
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.arguments ?? {}) },
    }));
}

function finishReason(result: GenerateResult<unknown>): string {
  return result.finishReason === 'tool-calls' ? 'tool_calls' : result.finishReason;
}

function usageOf(result: GenerateResult<unknown>): Record<string, number> | undefined {
  if (result.usage === undefined) return undefined;
  return {
    prompt_tokens: result.usage.inputTokens,
    completion_tokens: result.usage.outputTokens,
    total_tokens: result.usage.totalTokens,
  };
}

async function readBody(req: http.IncomingMessage, limit = 32 * 1024 * 1024): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > limit) throw new HttpError(413, 'Request body too large.');
    chunks.push(chunk as Buffer);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  if (text === '') return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, 'Request body is not valid JSON.');
  }
}

/** Build the server without listening. `serve()` is the one-liner. */
export function createServer(options: ServerOptions = {}): http.Server {
  const { port: _port, host: _host, tier: defaultTier = 'device', apiKey, cors, log, ...llmOptions } = options;
  const clients = new Map<string, AppleLLM>();
  const clientFor = (tier: 'device' | 'cloud' | 'auto'): AppleLLM => {
    let client = clients.get(tier);
    if (client === undefined) {
      client = new AppleLLM({ ...llmOptions, tier });
      clients.set(tier, client);
    }
    return client;
  };

  const server = http.createServer((req, res) => {
    const started = Date.now();
    const controller = new AbortController();
    // A client that hangs up mid-generation stops the generation too.
    res.on('close', () => {
      if (!res.writableFinished) controller.abort();
    });
    const send = (status: number, body: unknown): void => {
      if (res.headersSent) return;
      res.writeHead(status, { 'content-type': 'application/json', ...corsHeaders() });
      res.end(JSON.stringify(body));
    };
    const corsHeaders = (): Record<string, string> =>
      cors === undefined
        ? {}
        : {
            'access-control-allow-origin': cors,
            'access-control-allow-headers': 'authorization, content-type',
            'access-control-allow-methods': 'GET, POST, OPTIONS',
          };

    const route = async (): Promise<string | undefined> => {
      const url = new URL(req.url ?? '/', 'http://localhost');
      const pathname = url.pathname.replace(/\/+$/, '') || '/';
      if (req.method === 'OPTIONS') {
        res.writeHead(204, corsHeaders());
        res.end();
        return undefined;
      }
      if (apiKey !== undefined && req.headers.authorization !== `Bearer ${apiKey}`) {
        send(401, { error: { message: 'Invalid API key.', type: 'invalid_request_error', code: 'invalid_api_key', param: null } });
        return undefined;
      }
      if (req.method === 'GET' && (pathname === '/health' || pathname === '/v1/health')) {
        const state = await probe();
        send(state.device.available || state.cloud.available ? 200 : 503, {
          status: state.device.available || state.cloud.available ? 'ok' : 'unavailable',
          device: { available: state.device.available, variant: state.device.variant, contextSize: state.device.contextSize, reason: state.device.reason },
          cloud: { available: state.cloud.available, quota: state.cloud.quota?.status, reason: state.cloud.reason },
        });
        return undefined;
      }
      if (req.method === 'GET' && pathname === '/v1/models') {
        const created = Math.floor(Date.now() / 1000);
        send(200, {
          object: 'list',
          data: Object.keys(MODELS).map((id) => ({ id, object: 'model', created, owned_by: 'apple' })),
        });
        return undefined;
      }
      if (req.method === 'POST' && pathname === '/v1/chat/completions') {
        const body = (await readBody(req)) as ChatRequest;
        return chatCompletions(body);
      }
      send(404, { error: { message: `No route for ${req.method} ${pathname}.`, type: 'invalid_request_error', code: 'not_found', param: null } });
      return undefined;
    };

    const chatCompletions = async (body: ChatRequest): Promise<string> => {
      const requested = body.model ?? '';
      const tier = requested in MODELS ? MODELS[requested as keyof typeof MODELS] : defaultTier;
      const model = requested in MODELS ? requested : tier === 'cloud' ? 'apple-private-cloud' : 'apple-on-device';
      if (body.n !== undefined && body.n !== 1) throw new HttpError(400, 'Only n=1 is supported.');

      const images = new TempImages();
      try {
        const messages = await toMessages(body.messages, images, controller.signal);
        const functions: FunctionTool[] = [];
        if (body.tool_choice !== 'none') {
          for (const def of body.tools ?? []) {
            if (def.type !== 'function' || def.function === undefined) continue;
            functions.push(tool({ name: def.function.name, description: def.function.description, parameters: def.function.parameters }));
          }
        }
        let schema: JsonSchema | undefined;
        if (body.response_format?.type === 'json_schema' && body.response_format.json_schema?.schema !== undefined) {
          schema = body.response_format.json_schema.schema;
        } else if (body.response_format?.type === 'json_object') {
          messages.unshift({ role: 'system', content: 'Reply with a single JSON object and nothing else.' });
        }
        let sampling: SamplingMode | undefined;
        if (body.top_p !== undefined) sampling = { mode: 'threshold', p: body.top_p, seed: body.seed };
        else if (body.seed !== undefined) sampling = { mode: 'topK', k: 50, seed: body.seed };

        const call: TextOptions = {
          temperature: body.temperature,
          maxTokens: body.max_completion_tokens ?? body.max_tokens,
          sampling,
          tools: functions.length > 0 ? functions : undefined,
          signal: controller.signal,
        };
        const client = clientFor(tier);
        const id = `chatcmpl-${randomUUID().replace(/-/g, '').slice(0, 24)}`;
        const created = Math.floor(Date.now() / 1000);

        if (body.stream === true) {
          res.writeHead(200, {
            'content-type': 'text/event-stream',
            'cache-control': 'no-cache',
            connection: 'keep-alive',
            ...corsHeaders(),
          });
          // A client that hung up has a destroyed socket; the abort above has
          // already stopped generation, so there is nothing left to tell it.
          const event = (payload: unknown): void => {
            if (!res.destroyed) res.write(`data: ${typeof payload === 'string' ? payload : JSON.stringify(payload)}\n\n`);
          };
          const chunk = (delta: Record<string, unknown>, finish: string | null = null): void => {
            event({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta, finish_reason: finish, logprobs: null }] });
          };
          chunk({ role: 'assistant', content: '' });
          try {
            let result: GenerateResult<unknown>;
            if (schema === undefined) {
              const stream = client.stream(messages, call);
              for await (const delta of stream) chunk({ content: delta });
              result = await stream.result;
            } else {
              result = await client.generate(messages, { ...call, schema });
              if (result.finishReason !== 'tool-calls') chunk({ content: result.text });
            }
            const calls = openAIToolCalls(result.toolCalls);
            if (calls.length > 0 && result.finishReason === 'tool-calls') chunk({ tool_calls: calls });
            chunk({}, finishReason(result));
            if (body.stream_options?.include_usage === true) {
              event({ id, object: 'chat.completion.chunk', created, model, choices: [], usage: usageOf(result) ?? null });
            }
          } catch (error) {
            // Headers are gone; OpenAI clients read an error event in the stream.
            event(errorResponse(error).body);
            if (!res.destroyed) res.end('data: [DONE]\n\n');
            return `${model} stream failed: ${error instanceof Error ? error.message.split('\n')[0] : String(error)}`;
          }
          if (!res.destroyed) res.end('data: [DONE]\n\n');
          return `${model} stream`;
        }

        const result: GenerateResult<unknown> =
          schema === undefined ? await client.generate(messages, call) : await client.generate(messages, { ...call, schema });
        const calls = result.finishReason === 'tool-calls' ? openAIToolCalls(result.toolCalls) : [];
        send(200, {
          id,
          object: 'chat.completion',
          created,
          model,
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: result.finishReason === 'tool-calls' ? null : result.text,
                ...(calls.length > 0 ? { tool_calls: calls.map(({ index: _i, ...rest }) => rest) } : {}),
                refusal: null,
              },
              finish_reason: finishReason(result),
              logprobs: null,
            },
          ],
          usage: usageOf(result) ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        });
        return model;
      } finally {
        await images.dispose();
      }
    };

    route()
      .then((what) => {
        if (what !== undefined) log?.(`${req.method} ${req.url} ${what} ${res.statusCode} ${Date.now() - started}ms`);
      })
      .catch((error: unknown) => {
        const { status, body } = errorResponse(error);
        if (error instanceof ModelBusyError && !res.headersSent) res.setHeader('retry-after', '2');
        send(status, body);
        log?.(`${req.method} ${req.url} ${status} ${Date.now() - started}ms ${error instanceof Error ? error.message.split('\n')[0] : ''}`);
      });
  });

  server.on('close', () => {
    for (const client of clients.values()) client.close();
  });
  return server;
}

/** Start listening. Resolves once the port is bound. */
export async function serve(options: ServerOptions = {}): Promise<{ server: http.Server; url: string; close: () => Promise<void> }> {
  const server = createServer(options);
  const port = options.port ?? DEFAULT_PORT;
  const host = options.host ?? '127.0.0.1';
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      server.off('error', reject);
      resolve();
    });
  });
  const address = server.address();
  const bound = typeof address === 'object' && address !== null ? address.port : port;
  return {
    server,
    url: `http://${host === '0.0.0.0' ? '127.0.0.1' : host}:${bound}/v1`,
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections?.();
      }),
  };
}
