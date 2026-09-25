/**
 * Vercel AI SDK provider (specification v3: AI SDK 6 and 7).
 *
 *   import { generateText, streamText } from 'ai';
 *   import { apple } from 'apple-llm/ai-sdk';
 *
 *   const { text } = await generateText({ model: apple(), prompt: 'Hello' });
 *
 * `apple()` is the on-device model; `apple('cloud')` is Private Cloud Compute,
 * which sends the prompt off the machine; `apple('auto')` prefers the device.
 * Tools, structured output (`Output.object`), images, streaming and abort
 * signals all map onto the native features. The AI SDK runs tools itself, so
 * each call stops at a tool call and the SDK sends the result back — see
 * messages.ts for how that is replayed on a model that cannot resume from a
 * tool output.
 *
 * Only types are imported from `@ai-sdk/provider`; nothing here loads at
 * runtime unless you import this entry point.
 */
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3GenerateResult,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3ToolResultOutput,
  LanguageModelV3Usage,
  ProviderV3,
  SharedV3Warning,
} from '@ai-sdk/provider';
import { TempImages } from './attachments.js';
import { AppleLLM, type AppleLLMOptions, type GenerateResult, type TextOptions } from './client.js';
import type { BuiltInTool, FinishReason, Guardrails, SamplingMode, Usage, UseCase } from './device.js';
import { AppleLLMError } from './errors.js';
import type { ChatMessage } from './messages.js';
import type { JsonSchema } from './schema.js';
import { tool, type FunctionTool } from './tools.js';

export type AppleModelId = 'device' | 'cloud' | 'auto';

/** Per-call options, under `providerOptions: { apple: { … } }`. */
export interface AppleProviderOptions {
  useCase?: UseCase;
  guardrails?: Guardrails;
  /** Apple's built-in tools, run on device alongside your own. */
  builtInTools?: BuiltInTool[];
  /** Drop the oldest turns when a conversation outgrows the window. Default true. */
  trimHistory?: boolean;
  /** Cloud tier: "Use Broad World Knowledge". */
  webSearch?: boolean;
}

export interface AppleProviderSettings extends Omit<AppleLLMOptions, 'tier'> {
  /** Share one client (and so one warm helper) with the rest of your code. */
  client?: AppleLLM;
}

export interface AppleProvider extends ProviderV3 {
  (modelId?: AppleModelId): LanguageModelV3;
  /** `string` too, as ProviderV3 requires; anything but the three ids throws. */
  languageModel(modelId?: AppleModelId | (string & {})): LanguageModelV3;
}

const PROVIDER = 'apple';

function usageV3(usage: Usage | undefined): LanguageModelV3Usage {
  return {
    inputTokens: {
      total: usage?.inputTokens,
      noCache: usage === undefined ? undefined : usage.inputTokens - (usage.cachedInputTokens ?? 0),
      cacheRead: usage?.cachedInputTokens,
      cacheWrite: undefined,
    },
    outputTokens: { total: usage?.outputTokens, text: usage?.outputTokens, reasoning: undefined },
  };
}

function finishV3(reason: FinishReason): LanguageModelV3FinishReason {
  return { unified: reason, raw: reason };
}

function toolOutputText(output: LanguageModelV3ToolResultOutput): string {
  switch (output.type) {
    case 'text':
    case 'error-text':
      return output.value;
    case 'json':
    case 'error-json':
      return JSON.stringify(output.value);
    case 'execution-denied':
      return `The tool call was not executed${output.reason ? `: ${output.reason}` : '.'}`;
    case 'content':
      return output.value
        .map((part) => (part.type === 'text' ? part.text : `[${part.type}]`))
        .join('\n');
    default:
      return '';
  }
}

/** The SDK's prompt as this package's message list; images become temp files. */
async function toMessages(
  prompt: LanguageModelV3Prompt,
  images: TempImages,
  warnings: SharedV3Warning[],
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const out: ChatMessage[] = [];
  for (const message of prompt) {
    switch (message.role) {
      case 'system':
        out.push({ role: 'system', content: message.content });
        break;
      case 'user': {
        const texts: string[] = [];
        const paths: string[] = [];
        for (const part of message.content) {
          if (part.type === 'text') {
            texts.push(part.text);
          } else if (part.mediaType.startsWith('image/')) {
            paths.push(await images.add(part.data as never, part.mediaType, signal));
          } else if (part.mediaType.startsWith('text/') || part.mediaType === 'application/json') {
            const body =
              typeof part.data === 'string'
                ? Buffer.from(part.data, 'base64').toString('utf8')
                : part.data instanceof URL
                  ? await (await fetch(part.data, { signal })).text()
                  : Buffer.from(part.data).toString('utf8');
            texts.push(`--- Document: ${part.filename ?? 'attachment'} ---\n${body}`);
          } else {
            warnings.push({ type: 'unsupported', feature: `file part of type ${part.mediaType}` });
          }
        }
        out.push({ role: 'user', content: texts.join('\n\n'), ...(paths.length > 0 ? { images: paths } : {}) });
        break;
      }
      case 'assistant': {
        const texts: string[] = [];
        const calls: Array<{ id: string; name: string; arguments: unknown }> = [];
        const results: ChatMessage[] = [];
        for (const part of message.content) {
          if (part.type === 'text') texts.push(part.text);
          else if (part.type === 'tool-call') calls.push({ id: part.toolCallId, name: part.toolName, arguments: part.input });
          else if (part.type === 'tool-result') {
            results.push({ role: 'tool', toolCallId: part.toolCallId, name: part.toolName, content: toolOutputText(part.output) });
          }
        }
        out.push({ role: 'assistant', content: texts.join(''), ...(calls.length > 0 ? { toolCalls: calls } : {}) });
        out.push(...results);
        break;
      }
      case 'tool':
        for (const part of message.content) {
          if (part.type !== 'tool-result') continue;
          out.push({ role: 'tool', toolCallId: part.toolCallId, name: part.toolName, content: toolOutputText(part.output) });
        }
        break;
      default:
        break;
    }
  }
  return out;
}

class AppleLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3' as const;
  readonly provider = PROVIDER;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  constructor(
    readonly modelId: AppleModelId,
    private readonly client: AppleLLM,
  ) {}

  /** Map call options onto a request; collect warnings for what does not map. */
  private async prepare(options: LanguageModelV3CallOptions): Promise<{
    messages: ChatMessage[];
    call: TextOptions;
    schema?: JsonSchema;
    warnings: SharedV3Warning[];
    images: TempImages;
  }> {
    const warnings: SharedV3Warning[] = [];
    const images = new TempImages();
    const messages = await toMessages(options.prompt, images, warnings, options.abortSignal);
    const apple = (options.providerOptions?.apple ?? {}) as AppleProviderOptions;

    for (const [feature, value] of [
      ['stopSequences', options.stopSequences],
      ['presencePenalty', options.presencePenalty],
      ['frequencyPenalty', options.frequencyPenalty],
    ] as const) {
      if (value !== undefined && !(Array.isArray(value) && value.length === 0)) {
        warnings.push({ type: 'unsupported', feature });
      }
    }

    let sampling: SamplingMode | undefined;
    if (options.topK !== undefined) sampling = { mode: 'topK', k: options.topK, seed: options.seed };
    else if (options.topP !== undefined) sampling = { mode: 'threshold', p: options.topP, seed: options.seed };
    else if (options.seed !== undefined) sampling = { mode: 'topK', k: 50, seed: options.seed };

    const functions: FunctionTool[] = [];
    if (options.toolChoice?.type !== 'none') {
      for (const def of options.tools ?? []) {
        if (def.type !== 'function') {
          warnings.push({ type: 'unsupported', feature: `provider tool ${def.name}` });
          continue;
        }
        // No execute: the SDK runs the tool and calls back with the result.
        functions.push(tool({ name: def.name, description: def.description, parameters: def.inputSchema as JsonSchema }));
      }
    }
    if (options.toolChoice?.type === 'required' || options.toolChoice?.type === 'tool') {
      warnings.push({
        type: 'unsupported',
        feature: `toolChoice "${options.toolChoice.type}"`,
        details: 'Apple’s model decides for itself whether to call a tool.',
      });
    }

    let schema: JsonSchema | undefined;
    if (options.responseFormat?.type === 'json') {
      if (options.responseFormat.schema !== undefined) {
        schema = options.responseFormat.schema as JsonSchema;
      } else {
        // Constrained decoding needs a schema; without one, JSON is only requested.
        messages.unshift({ role: 'system', content: 'Reply with a single JSON value and nothing else.' });
        warnings.push({ type: 'compatibility', feature: 'responseFormat json without a schema', details: 'requested in the prompt, not enforced' });
      }
    }

    const tools = [...(apple.builtInTools ?? []), ...functions];
    return {
      messages,
      schema,
      warnings,
      images,
      call: {
        temperature: options.temperature,
        maxTokens: options.maxOutputTokens,
        sampling,
        signal: options.abortSignal,
        tools: tools.length > 0 ? tools : undefined,
        useCase: apple.useCase,
        guardrails: apple.guardrails,
        trimHistory: apple.trimHistory,
        webSearch: apple.webSearch,
      },
    };
  }

  /** Content parts: the text, then any tool call the SDK has to run. */
  private content(result: GenerateResult<unknown>): LanguageModelV3Content[] {
    const content: LanguageModelV3Content[] = [];
    if (result.text !== '' && result.finishReason !== 'tool-calls') content.push({ type: 'text', text: result.text });
    for (const call of result.toolCalls) {
      // Calls already answered (replayed, or Apple's built-ins) are history.
      if (call.output !== undefined) continue;
      content.push({ type: 'tool-call', toolCallId: call.id, toolName: call.name, input: JSON.stringify(call.arguments ?? {}) });
    }
    return content;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const { messages, call, schema, warnings, images } = await this.prepare(options);
    try {
      const result: GenerateResult<unknown> =
        schema === undefined
          ? await this.client.generate(messages, call)
          : await this.client.generate(messages, { ...call, schema });
      return {
        content: this.content(result),
        finishReason: finishV3(result.finishReason),
        usage: usageV3(result.usage),
        warnings,
        response: { id: `apple-${Date.now().toString(36)}`, timestamp: new Date(), modelId: this.modelId },
        providerMetadata: {
          apple: { tier: result.tier, durationMs: result.durationMs, trimmedTurns: result.trimmedTurns },
        },
      };
    } finally {
      await images.dispose();
    }
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const { messages, call, schema, warnings, images } = await this.prepare(options);
    const client = this.client;
    const modelId = this.modelId;
    const content = (result: GenerateResult<unknown>): LanguageModelV3Content[] => this.content(result);

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: 'stream-start', warnings });
        controller.enqueue({ type: 'response-metadata', id: `apple-${Date.now().toString(36)}`, timestamp: new Date(), modelId });
        const textId = 'text-0';
        let textOpen = false;
        const openText = (): void => {
          if (!textOpen) controller.enqueue({ type: 'text-start', id: textId });
          textOpen = true;
        };
        try {
          let result: GenerateResult<unknown>;
          if (schema === undefined) {
            const textStream = client.stream(messages, call);
            for await (const delta of textStream) {
              openText();
              controller.enqueue({ type: 'text-delta', id: textId, delta });
            }
            result = await textStream.result;
          } else {
            // Partial snapshots are whole documents whose key order can shift,
            // so they are not text deltas; the SDK gets the final JSON once.
            result = await client.generate(messages, { ...call, schema });
            if (result.finishReason !== 'tool-calls') {
              openText();
              controller.enqueue({ type: 'text-delta', id: textId, delta: result.text });
            }
          }
          if (textOpen) controller.enqueue({ type: 'text-end', id: textId });
          for (const part of content(result)) {
            if (part.type === 'tool-call') controller.enqueue(part);
          }
          controller.enqueue({
            type: 'finish',
            finishReason: finishV3(result.finishReason),
            usage: usageV3(result.usage),
            providerMetadata: { apple: { tier: result.tier, durationMs: result.durationMs, trimmedTurns: result.trimmedTurns } },
          });
        } catch (error) {
          controller.enqueue({ type: 'error', error });
        } finally {
          await images.dispose();
          controller.close();
        }
      },
    });
    return { stream };
  }
}

/**
 * Create a provider. Pass settings to change defaults for every model it
 * makes, or `client` to share an existing AppleLLM (and its warm helper).
 */
export function createApple(settings: AppleProviderSettings = {}): AppleProvider {
  const { client: shared, ...defaults } = settings;
  const clients = new Map<AppleModelId, AppleLLM>();
  const clientFor = (tier: AppleModelId): AppleLLM => {
    if (shared !== undefined) return shared;
    let client = clients.get(tier);
    if (client === undefined) {
      client = new AppleLLM({ ...defaults, tier });
      clients.set(tier, client);
    }
    return client;
  };
  const languageModel = (modelId: string = 'device'): LanguageModelV3 => {
    if (modelId !== 'device' && modelId !== 'cloud' && modelId !== 'auto') {
      throw new AppleLLMError(`Unknown Apple model "${String(modelId)}" (want device, cloud or auto).`);
    }
    return new AppleLanguageModel(modelId, clientFor(modelId));
  };
  const unsupported = (kind: string) => (): never => {
    throw new AppleLLMError(`Apple's built-in models do not provide ${kind} models.`);
  };
  return Object.assign((modelId?: AppleModelId) => languageModel(modelId), {
    specificationVersion: 'v3' as const,
    languageModel,
    embeddingModel: unsupported('embedding'),
    imageModel: unsupported('image'),
  }) as AppleProvider;
}

/** The default provider: `apple()` on device, `apple('cloud')`, `apple('auto')`. */
export const apple: AppleProvider = createApple();
