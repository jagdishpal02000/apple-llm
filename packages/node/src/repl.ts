/**
 * `apple-llm chat` — an interactive conversation in the terminal.
 *
 * Replies stream as they are generated. Ctrl+C stops the reply in progress
 * (and only that); Ctrl+C at an empty prompt, Ctrl+D or /exit quits. The
 * conversation is kept as a message list and trimmed from the oldest end when
 * it outgrows the context window, with a note when that happens.
 */
import { createInterface } from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { AbortError } from './errors.js';
import type { AppleLLM, TextOptions } from './client.js';
import type { ImageAttachment } from './device.js';
import type { ChatMessage } from './messages.js';

const HELP = `commands:
  /reset            start over
  /system <text>    replace the instructions (and start over)
  /image <path>     attach an image to your next message
  /tokens           how much of the context window the conversation uses
  /help             this list
  /exit             quit (or Ctrl+D)`;

export interface ReplOptions {
  system?: string;
  call?: TextOptions;
}

export async function runRepl(llm: AppleLLM, options: ReplOptions = {}): Promise<void> {
  const tty = output.isTTY === true;
  const dim = (s: string): string => (tty ? `\x1b[2m${s}\x1b[22m` : s);
  const rl = createInterface({ input, output, terminal: tty });
  // Created before anything is awaited: the iterator buffers lines, so input
  // piped in while the model warms up is not lost (rl.question would drop it).
  const lines = rl[Symbol.asyncIterator]();
  rl.setPrompt(tty ? '\x1b[1myou ›\x1b[22m ' : 'you › ');
  let system = options.system;
  let messages: ChatMessage[] = [];
  let pendingImages: ImageAttachment[] = [];
  let generating: AbortController | undefined;

  // Ctrl+C stops a reply; at the prompt it quits. Closing ends the line
  // iterator, which is what ends the loop — after any lines already buffered.
  rl.on('SIGINT', () => {
    if (generating !== undefined) generating.abort();
    else rl.close();
  });

  await llm.ensureReady((p) => output.write(dim(`  ${p.status}\n`)));
  output.write(dim(`${llm.label} — ${llm.contextSize ?? '?'} token context. /help for commands, Ctrl+D to quit.\n`));

  for (;;) {
    if (tty) rl.prompt();
    const next = await lines.next();
    if (next.done === true) break;
    const line = String(next.value).trim();
    // Piped input is not echoed by a terminal; echo it so the output reads as a transcript.
    if (!tty) output.write(`you › ${line}\n`);
    if (line === '') continue;

    if (line.startsWith('/')) {
      const [command, ...args] = line.slice(1).split(/\s+/);
      const rest = args.join(' ');
      if (command === 'exit' || command === 'quit') break;
      if (command === 'help') output.write(`${HELP}\n`);
      else if (command === 'reset') {
        messages = [];
        output.write(dim('conversation cleared\n'));
      } else if (command === 'system') {
        system = rest === '' ? undefined : rest;
        messages = [];
        output.write(dim(system === undefined ? 'instructions cleared\n' : 'instructions set; conversation cleared\n'));
      } else if (command === 'image') {
        if (rest === '') output.write('usage: /image <path>\n');
        else {
          pendingImages.push(rest);
          output.write(dim(`attached ${rest} to your next message\n`));
        }
      } else if (command === 'tokens') {
        if (messages.length === 0) output.write(dim('nothing yet\n'));
        else {
          try {
            const probe: ChatMessage[] = [...messages, { role: 'user', content: '' }];
            const { tokens, contextSize } = await llm.countTokens(probe, { system });
            output.write(dim(`${tokens} of ${contextSize} tokens (${Math.round((tokens / contextSize) * 100)}%)\n`));
          } catch (error) {
            output.write(`${error instanceof Error ? error.message : String(error)}\n`);
          }
        }
      } else output.write(`unknown command /${command}\n${HELP}\n`);
      continue;
    }

    const turn: ChatMessage = { role: 'user', content: line, ...(pendingImages.length > 0 ? { images: pendingImages } : {}) };
    pendingImages = [];
    generating = new AbortController();
    output.write(tty ? '\x1b[1mmodel ›\x1b[22m ' : 'model › ');
    try {
      const stream = llm.stream([...messages, turn], { ...options.call, system, signal: generating.signal });
      for await (const delta of stream) output.write(delta);
      const result = await stream.result;
      output.write('\n');
      messages.push(turn, result.message);
      const notes: string[] = [];
      if (result.trimmedTurns > 0) notes.push(`${result.trimmedTurns} oldest turn(s) dropped to fit`);
      if (result.finishReason === 'length') notes.push('cut off at the token limit');
      if (result.usage !== undefined) notes.push(`${result.usage.outputTokens} tokens`);
      notes.push(`${(result.durationMs / 1000).toFixed(1)}s`);
      output.write(dim(`  ${notes.join(' · ')}\n`));
    } catch (error) {
      output.write('\n');
      if (error instanceof AbortError) output.write(dim('  (stopped)\n'));
      else output.write(`${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}\n`);
    } finally {
      generating = undefined;
    }
  }
  rl.close();
}
