/**
 * OpenAI-style message lists, mapped onto what Apple's model actually has.
 *
 * Apple has one instructions slot, a native transcript of prompt / response /
 * tool-call / tool-output entries, and one new prompt per call. So: every
 * system message joins the instructions, the last user message is the prompt,
 * and everything before it becomes a native transcript — not a text preamble,
 * which the model would read as part of the question.
 *
 * Messages *after* the last user message are a tool exchange in progress: the
 * model asked for tool calls, the caller ran them, and the results are here.
 * Apple cannot resume generation from a tool output without a new prompt, so
 * the exchange is replayed instead: the same prompt runs again, and when the
 * model makes the same call its recorded result is returned immediately rather
 * than executed twice.
 */
import type { ImageAttachment } from './device.js';
import { AppleLLMError } from './errors.js';

export interface ChatToolCall {
  id: string;
  name: string;
  arguments: unknown;
}

export type ChatMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string; images?: ImageAttachment[] }
  | { role: 'assistant'; content?: string | null; toolCalls?: ChatToolCall[] }
  | { role: 'tool'; toolCallId: string; name?: string; content: string };

/** The helper's wire form of one history entry. */
export type HistoryEntry =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: ChatToolCall[] }
  | { role: 'tool'; toolCallId: string; name: string; content: string };

/** A tool result recorded by the caller, to be handed back if the model repeats the call. */
export interface ReplayedResult {
  name: string;
  arguments: unknown;
  output: string;
}

export interface SplitMessages {
  system?: string;
  history: HistoryEntry[];
  prompt: string;
  /** Images on the last user message, which the model sees. */
  images: ImageAttachment[];
  replay: ReplayedResult[];
}

/** Map message lists onto instructions + history + prompt (+ replayed tool results). */
export function splitMessages(messages: ReadonlyArray<ChatMessage>): SplitMessages {
  const systems: string[] = [];
  const rest: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role === 'system') systems.push(message.content);
    else rest.push(message);
  }
  let lastUser = -1;
  for (let i = rest.length - 1; i >= 0; i -= 1) {
    if (rest[i].role === 'user') {
      lastUser = i;
      break;
    }
  }
  if (lastUser === -1) {
    throw new AppleLLMError('A message list needs at least one user message.');
  }
  const tail = rest.slice(lastUser + 1);
  const final = tail[tail.length - 1];
  if (final !== undefined && final.role !== 'tool') {
    throw new AppleLLMError(
      'The last message must be from the user, or a tool result answering the assistant’s tool call.',
    );
  }

  const callNames = new Map<string, ChatToolCall>();
  for (const message of rest) {
    if (message.role === 'assistant') for (const call of message.toolCalls ?? []) callNames.set(call.id, call);
  }

  const history: HistoryEntry[] = [];
  for (const message of rest.slice(0, lastUser)) {
    const entry = toHistory(message, callNames);
    if (entry !== undefined) history.push(entry);
  }

  const replay: ReplayedResult[] = [];
  for (const message of tail) {
    if (message.role !== 'tool') continue;
    const call = callNames.get(message.toolCallId);
    replay.push({
      name: message.name ?? call?.name ?? '',
      arguments: call?.arguments ?? {},
      output: message.content,
    });
  }

  const last = rest[lastUser] as Extract<ChatMessage, { role: 'user' }>;
  return {
    system: systems.length > 0 ? systems.join('\n\n') : undefined,
    history,
    prompt: last.content,
    images: last.images ?? [],
    replay,
  };
}

function toHistory(message: ChatMessage, calls: Map<string, ChatToolCall>): HistoryEntry | undefined {
  switch (message.role) {
    case 'user': {
      // A native transcript entry carries text only here, so an earlier image
      // is noted rather than dropped without a trace.
      const images = message.images?.length ?? 0;
      const note = images === 0 ? '' : ` [${images === 1 ? 'an image was' : `${images} images were`} attached here]`;
      return { role: 'user', content: `${message.content}${note}` };
    }
    case 'assistant':
      return {
        role: 'assistant',
        content: message.content ?? '',
        ...(message.toolCalls !== undefined && message.toolCalls.length > 0 ? { toolCalls: message.toolCalls } : {}),
      };
    case 'tool':
      return {
        role: 'tool',
        toolCallId: message.toolCallId,
        name: message.name ?? calls.get(message.toolCallId)?.name ?? 'tool',
        content: message.content,
      };
    default:
      return undefined;
  }
}

/**
 * History as plain text, for the cloud tier: Shortcuts takes one string, so a
 * transcript preamble is the only way to carry earlier turns there.
 */
export function renderHistory(history: ReadonlyArray<HistoryEntry>): string {
  return history
    .map((entry) => {
      if (entry.role === 'user') return `User: ${entry.content}`;
      if (entry.role === 'tool') return `Tool ${entry.name} returned: ${entry.content}`;
      const calls = (entry.toolCalls ?? []).map((c) => `[called ${c.name}(${JSON.stringify(c.arguments)})]`);
      return `Assistant: ${[entry.content, ...calls].filter((s) => s !== '').join(' ')}`;
    })
    .join('\n');
}

/** Stable text for comparing tool arguments regardless of key order. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson((value as Record<string, unknown>)[k])}`).join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/**
 * Hands recorded results back to repeated calls: an exact (name, arguments)
 * match first, then the next unused result for that tool name — the model's
 * second attempt at "Paris, France" should still get the Paris weather.
 */
export class ReplayBook {
  private readonly unused: ReplayedResult[];

  constructor(results: ReadonlyArray<ReplayedResult>) {
    this.unused = [...results];
  }

  take(name: string, args: unknown): string | undefined {
    const key = canonicalJson(args);
    let index = this.unused.findIndex((r) => r.name === name && canonicalJson(r.arguments) === key);
    if (index < 0) index = this.unused.findIndex((r) => r.name === name);
    if (index < 0) return undefined;
    const [hit] = this.unused.splice(index, 1);
    return hit.output;
  }
}
