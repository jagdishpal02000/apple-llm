import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { AbortError, AppleLLMError, TimeoutError } from './errors.js';

/** How a child is created. Injectable so tests can script stdout without Apple hardware. */
export type Spawner = (binary: string, args: string[]) => ChildProcess;

const defaultSpawner: Spawner = (binary, args) =>
  spawn(binary, args, { stdio: ['pipe', 'pipe', 'pipe'] });

export interface HelperServerOptions {
  timeoutMs?: number;
  spawner?: Spawner;
}

/** A line the helper sent before a request's final line (`"done": false`). */
export type HelperEvent = Record<string, unknown> & { done: false };

export interface RequestHooks {
  /** Called for every event line; `write` sends a control line for this request. */
  onEvent?: (event: HelperEvent, write: (control: Record<string, unknown>) => void) => void;
  /** Cancels the request: a queued one is never sent, a running one is cancelled in the helper. */
  signal?: AbortSignal;
  /** Request id, the target of a cancel line. Needed to cancel mid-flight. */
  id?: string;
}

interface Pending {
  hooks: RequestHooks;
  timer?: NodeJS.Timeout;
  /** The caller gave up; the helper's reply is still consumed to keep pairing intact. */
  abandoned: boolean;
  /** Completes the request: frees its slot, and answers the caller unless abandoned. */
  finish: (line: string | undefined, err?: Error) => void;
}

/** What `signal.reason` means to a caller: a timeout stays a timeout. */
export function abortErrorFrom(signal: AbortSignal): Error {
  const reason: unknown = signal.reason;
  if (reason instanceof AppleLLMError) return reason;
  if (reason instanceof Error && reason.name === 'TimeoutError') {
    return new TimeoutError(`Apple generation timed out: ${reason.message}`, 'device', { cause: reason });
  }
  return new AbortError('The request was aborted.', undefined, { cause: reason });
}

/**
 * A long-lived helper process speaking newline-delimited JSON.
 *
 * This is the single most important piece of the on-device path. Spawning a
 * helper per request measured ~17s per call against ~1.5s once the model was
 * resident, with identical prompts and identically short outputs — so keeping
 * one process alive is worth roughly an order of magnitude. Both failure modes
 * it guards against are silent: they produce correct output, just 10-20x
 * slower.
 *
 * Requests are serialised, and that costs nothing: Apple's framework serialises
 * inference anyway. Four concurrent requests measured 29.19s against 29.45s
 * sequentially, so concurrency > 1 buys precisely nothing. A request owns the
 * head of the queue from the moment it is written until its final line; the
 * lines in between (`"done": false`) are its events — stream deltas, partial
 * objects, tool calls — and nothing else is written meanwhile except control
 * lines for that same request.
 *
 * The child is unref'd whenever nothing is in flight, so a script that forgets
 * `close()` still exits instead of hanging on an idle helper.
 */
export class HelperServer {
  private child: ChildProcess | undefined;
  private buffer = '';
  private queue: Pending[] = [];
  /** Serialises callers so one request's reply cannot be handed to another. */
  private chain: Promise<unknown> = Promise.resolve();
  private inFlight = 0;
  /** The helper's recent stderr, quoted when it dies. Always drained: an unread pipe blocks the writer. */
  private stderrTail = '';
  private readonly timeoutMs: number;
  private readonly spawner: Spawner;

  constructor(private readonly binary: string, options: HelperServerOptions = {}) {
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.spawner = options.spawner ?? defaultSpawner;
  }

  private start(): ChildProcess {
    if (this.child !== undefined && this.child.exitCode === null && !this.child.killed) {
      return this.child;
    }
    const child = this.spawner(this.binary, ['--serve']);
    this.child = child;
    this.buffer = '';
    this.stderrTail = '';

    child.stdout?.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString();
      for (;;) {
        const newline = this.buffer.indexOf('\n');
        if (newline === -1) break;
        const line = this.buffer.slice(0, newline).trim();
        this.buffer = this.buffer.slice(newline + 1);
        if (line !== '') this.dispatch(line);
      }
    });
    child.stderr?.on('data', (chunk: Buffer) => {
      this.stderrTail = (this.stderrTail + chunk.toString()).slice(-2000);
    });

    const fail = (err: Error): void => {
      if (this.child === child) this.child = undefined;
      const pending = this.queue;
      this.queue = [];
      for (const p of pending) {
        clearTimeout(p.timer);
        p.finish(undefined, err);
      }
    };
    child.on('error', (err) => fail(err instanceof Error ? err : new Error(String(err))));
    child.on('close', (code) => {
      const tail = this.stderrTail.trim();
      fail(new AppleLLMError(`Apple helper exited with code ${code}${tail ? `: ${tail.slice(-400)}` : ''}`, 'device'));
    });
    child.stdin?.on('error', () => {
      /* close/error handlers report the real failure */
    });
    // Never let the helper outlive this process, however the run ends.
    const killer = (): void => {
      if (child.exitCode === null) child.kill();
    };
    process.once('exit', killer);
    child.once('close', () => process.removeListener('exit', killer));
    this.setRef(this.inFlight > 0);
    return child;
  }

  /** Keep the event loop alive only while a request is in flight. */
  private setRef(on: boolean): void {
    const child = this.child;
    if (child === undefined) return;
    const handles = [child, child.stdin, child.stdout, child.stderr] as Array<
      { ref?: () => void; unref?: () => void } | null | undefined
    >;
    for (const handle of handles) {
      if (on) handle?.ref?.();
      else handle?.unref?.();
    }
  }

  /** Route one line: an event to the head's hooks, anything else completes the head. */
  private dispatch(line: string): void {
    const head = this.queue[0];
    if (head === undefined) return;
    let parsed: Record<string, unknown> | undefined;
    try {
      const value: unknown = JSON.parse(line);
      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        parsed = value as Record<string, unknown>;
      }
    } catch {
      // Not JSON: complete the request with the raw line so the caller sees it.
    }
    if (parsed?.done === false) {
      this.arm(head);
      if (head.abandoned) {
        // Nobody is listening, but a tool call parks the helper until it is
        // answered; stop it rather than leave generation waiting forever.
        const call = parsed.toolCall as { id?: unknown } | undefined;
        if (typeof call?.id === 'string') this.write({ op: 'toolResult', callId: call.id, stop: true });
        return;
      }
      try {
        head.hooks.onEvent?.(parsed as HelperEvent, (control) => this.write(control));
      } catch {
        // A throwing callback must not break the framing for later requests.
      }
      return;
    }
    this.queue.shift();
    clearTimeout(head.timer);
    head.finish(line);
  }

  /**
   * (Re)start the wedge timer. It measures silence, not total time: every
   * event resets it, so a long stream is fine while a helper that stops
   * talking is not.
   */
  private arm(entry: Pending): void {
    clearTimeout(entry.timer);
    entry.timer = setTimeout(() => this.timeOut(entry), this.timeoutMs);
  }

  /**
   * Pause the wedge timer while the client itself is busy — a function tool
   * running in JS. The helper is legitimately silent then: it is waiting on us.
   */
  pause(): void {
    const head = this.queue[0];
    if (head !== undefined) clearTimeout(head.timer);
  }

  /** Resume the wedge timer paused by `pause()`. */
  resume(): void {
    const head = this.queue[0];
    if (head !== undefined) this.arm(head);
  }

  private timeOut(entry: Pending): void {
    const index = this.queue.indexOf(entry);
    if (index < 0) return; // already completed by the data handler
    this.queue.splice(index, 1);
    // A wedged helper cannot be trusted to stay in sync; drop it so the
    // next request starts from a clean process rather than reading a
    // reply that belongs to the request that timed out.
    this.stop();
    entry.finish(
      undefined,
      new TimeoutError(`Apple on-device generation timed out after ${this.timeoutMs / 1000}s of silence.`, 'device'),
    );
  }

  /** Write a control line (cancel, toolResult) for the request in flight. */
  write(control: Record<string, unknown>): void {
    this.child?.stdin?.write(`${JSON.stringify(control)}\n`);
  }

  /**
   * One request. Resolves with its final line; events go to `hooks.onEvent`.
   * Serialised against every other request through one chain.
   */
  request(payload: string, hooks: RequestHooks = {}): Promise<string> {
    const { signal } = hooks;
    if (signal?.aborted === true) return Promise.reject(abortErrorFrom(signal));

    let entry: Pending | undefined;
    let rejectCaller: (err: Error) => void = () => undefined;

    const caller = new Promise<string>((resolve, reject) => {
      rejectCaller = reject;
      const run = (): Promise<void> => {
        if (signal?.aborted === true) {
          // Aborted while queued: never written, so there is no reply to consume.
          reject(abortErrorFrom(signal));
          return Promise.resolve();
        }
        const child = this.start();
        this.inFlight += 1;
        this.setRef(true);
        return new Promise<void>((settle) => {
          const pending: Pending = {
            hooks,
            abandoned: false,
            finish: (line, err) => {
              this.inFlight -= 1;
              if (this.inFlight === 0) this.setRef(false);
              settle();
              if (pending.abandoned) return;
              if (err !== undefined) reject(err);
              else resolve(line ?? '');
            },
          };
          entry = pending;
          this.arm(pending);
          this.queue.push(pending);
          // JSON.stringify escapes newlines inside strings, so this only guards
          // against a caller handing us a pre-built payload with a literal one.
          child.stdin?.write(`${payload.replace(/\n/g, ' ')}\n`);
        });
      };
      // Chain regardless of whether the previous call settled or threw.
      const slot = this.chain.then(run, run);
      this.chain = slot.catch(() => undefined);
    });

    if (signal !== undefined) {
      const onAbort = (): void => {
        if (entry !== undefined && this.queue.includes(entry)) {
          // In flight: ask the helper to stop, and stop listening. The reply —
          // cancelled, or finished anyway — still completes this entry, so the
          // next request cannot be handed it.
          entry.abandoned = true;
          if (hooks.id !== undefined) this.write({ op: 'cancel', id: hooks.id });
        }
        rejectCaller(abortErrorFrom(signal));
      };
      signal.addEventListener('abort', onAbort, { once: true });
      const detach = (): void => signal.removeEventListener('abort', onAbort);
      caller.then(detach, detach);
    }
    return caller;
  }

  /** Version-1 shape: one request, one line. */
  send(payload: string, hooks: RequestHooks = {}): Promise<string> {
    return this.request(payload, hooks);
  }

  /**
   * Streaming request, kept for callers of the version-1 API: deltas go to
   * `onDelta` and the promise resolves with the final line.
   */
  stream(payload: string, onDelta: (delta: string) => void, hooks: RequestHooks = {}): Promise<string> {
    return this.request(payload, {
      ...hooks,
      onEvent: (event, write) => {
        if (typeof event.delta === 'string') onDelta(event.delta);
        hooks.onEvent?.(event, write);
      },
    });
  }

  stop(): void {
    const child = this.child;
    this.child = undefined;
    if (child !== undefined && child.exitCode === null) {
      child.stdin?.end();
      child.kill();
    }
  }
}
