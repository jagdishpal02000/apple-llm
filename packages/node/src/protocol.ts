import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import { TimeoutError } from './errors.js';

/** How a child is created. Injectable so tests can script stdout without Apple hardware. */
export type Spawner = (binary: string, args: string[]) => ChildProcess;

const defaultSpawner: Spawner = (binary, args) =>
  spawn(binary, args, { stdio: ['pipe', 'pipe', 'pipe'] });

export interface HelperServerOptions {
  timeoutMs?: number;
  spawner?: Spawner;
}

/**
 * A long-lived helper process handling newline-delimited JSON.
 *
 * This is the single most important piece of the on-device path. api-scribe
 * measured ~17s per call when it spawned a helper per request against ~1.5s once
 * the model was resident, with identical prompts and identically short outputs —
 * so keeping one process alive is worth roughly an order of magnitude. Both
 * failure modes it guards against are silent: they produce correct output, just
 * 10-20x slower.
 *
 * Requests are serialised, and that costs nothing: Apple's framework serialises
 * inference anyway. Four concurrent requests measured 29.19s against 29.45s
 * sequentially, so concurrency > 1 buys precisely nothing.
 */
export class HelperServer {
  private child: ChildProcess | undefined;
  private buffer = '';
  private queue: Array<{
    resolve: (line: string) => void;
    reject: (err: Error) => void;
    timer: NodeJS.Timeout;
    streaming?: boolean;
    onDelta?: (delta: string) => void;
    parts?: string[];
  }> = [];
  /** Serialises callers so one request's reply cannot be handed to another. */
  private chain: Promise<unknown> = Promise.resolve();
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

    child.stdout?.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString();
      // One response per line, in the order requests were written — except a
      // streaming request, which owns the head of the queue across N delta
      // lines until its final done:true line. The next request is not written
      // until the stream completes (see stream()), so replies cannot interleave.
      for (;;) {
        const newline = this.buffer.indexOf('\n');
        if (newline === -1) break;
        const line = this.buffer.slice(0, newline).trim();
        this.buffer = this.buffer.slice(newline + 1);
        if (line === '') continue;
        const head = this.queue[0];
        if (head === undefined) continue;
        if (head.streaming === true) {
          let parsedOk = false;
          let parsedDone = false;
          let parsedDelta = '';
          try {
            const parsed = JSON.parse(line) as { ok?: unknown; delta?: unknown; done?: unknown };
            parsedOk = parsed.ok === true;
            parsedDone = parsed.done === true;
            parsedDelta = typeof parsed.delta === 'string' ? parsed.delta : '';
          } catch {
            // Not JSON — fall through to complete the request with the raw line.
          }
          if (parsedOk && !parsedDone) {
            // Delta line: forward, refresh the wedge timer, keep waiting.
            clearTimeout(head.timer);
            head.timer = setTimeout(() => this.timeOut(head), this.timeoutMs);
            try {
              head.onDelta?.(parsedDelta);
            } catch {
              // A throwing onDelta must not break the stream.
            }
            continue;
          }
          // Final line (done:true) or an error envelope: complete the request.
          this.queue.shift();
          clearTimeout(head.timer);
          head.resolve(line);
          continue;
        }
        const pending = this.queue.shift();
        if (pending === undefined) continue;
        clearTimeout(pending.timer);
        pending.resolve(line);
      }
    });

    const fail = (err: Error): void => {
      this.child = undefined;
      const pending = this.queue;
      this.queue = [];
      for (const p of pending) {
        clearTimeout(p.timer);
        p.reject(err);
      }
    };
    child.on('error', (err) => fail(err instanceof Error ? err : new Error(String(err))));
    child.on('close', (code) => fail(new Error(`Apple helper exited with code ${code}`)));
    child.stdin?.on('error', () => {
      /* close/error handlers report the real failure */
    });
    // Never let the helper outlive this process, however the run ends.
    const killer = (): void => {
      if (child.exitCode === null) child.kill();
    };
    process.once('exit', killer);
    child.once('close', () => process.removeListener('exit', killer));
    return child;
  }

  private timeOut(entry: { timer: NodeJS.Timeout; reject: (e: Error) => void }): void {
    const index = this.queue.findIndex((p) => p.timer === entry.timer);
    if (index < 0) return; // already completed by the data handler
    this.queue.splice(index, 1);
    // A wedged helper cannot be trusted to stay in sync; drop it so the
    // next request starts from a clean process rather than reading a
    // reply that belongs to the request that timed out.
    this.stop();
    const err = new TimeoutError(
      `Apple on-device generation timed out after ${this.timeoutMs / 1000}s.`,
      'device',
    );
    entry.reject(err);
  }

  send(payload: string): Promise<string> {
    const run = async (): Promise<string> => {
      const child = this.start();
      return await new Promise<string>((resolve, reject) => {
        const entry = { resolve, reject, timer: undefined as unknown as NodeJS.Timeout };
        entry.timer = setTimeout(() => {
          const index = this.queue.findIndex((p) => p.timer === entry.timer);
          if (index >= 0) this.queue.splice(index, 1);
          // A wedged helper cannot be trusted to stay in sync; drop it so the
          // next request starts from a clean process rather than reading a
          // reply that belongs to the request that timed out.
          this.stop();
          reject(new TimeoutError(`Apple on-device generation timed out after ${this.timeoutMs / 1000}s.`, 'device'));
        }, this.timeoutMs);
        this.queue.push(entry);
        // JSON.stringify escapes newlines inside strings, so this only guards
        // against a caller handing us a pre-built payload with a literal one.
        child.stdin?.write(`${payload.replace(/\n/g, ' ')}\n`);
      });
    };
    // Chain regardless of whether the previous call settled or threw.
    const next = this.chain.then(run, run);
    this.chain = next.catch(() => undefined);
    return next;
  }

  /**
   * Streaming request. The helper emits N delta lines plus a final done:true
   * line; each delta is forwarded to onDelta as it arrives and the promise
   * resolves with the final line. Serialised against send() through the same
   * chain, so a stream never interleaves with another request.
   */
  stream(payload: string, onDelta: (delta: string) => void): Promise<string> {
    const run = async (): Promise<string> => {
      const child = this.start();
      return await new Promise<string>((resolve, reject) => {
        const entry = {
          resolve,
          reject,
          timer: undefined as unknown as NodeJS.Timeout,
          streaming: true as const,
          onDelta,
          parts: [] as string[],
        };
        entry.timer = setTimeout(() => this.timeOut(entry), this.timeoutMs);
        this.queue.push(entry);
        child.stdin?.write(`${payload.replace(/\n/g, ' ')}\n`);
      });
    };
    const next = this.chain.then(run, run);
    this.chain = next.catch(() => undefined);
    return next;
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
