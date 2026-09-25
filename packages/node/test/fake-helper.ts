import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import type { ChildProcess } from 'node:child_process';
import { HelperServer } from '../src/protocol.js';

/**
 * A scripted stand-in for the Swift helper speaking protocol v2, so the event
 * framing, control lines and tool loop can be tested on any machine.
 */
export class FakeChild extends EventEmitter {
  stdin = new PassThrough();
  stdout = new PassThrough();
  stderr = new PassThrough();
  exitCode: number | null = null;
  killed = false;
  written: Array<Record<string, unknown>> = [];
  refs = 0;
  unrefs = 0;

  constructor(private readonly onLine?: (message: Record<string, unknown>, child: FakeChild) => void) {
    super();
    this.stdin.on('data', (chunk: Buffer) => {
      for (const line of chunk.toString().split('\n')) {
        if (line.trim() === '') continue;
        const message = JSON.parse(line) as Record<string, unknown>;
        this.written.push(message);
        this.onLine?.(message, this);
      }
    });
  }

  reply(message: Record<string, unknown>): void {
    this.stdout.write(`${JSON.stringify(message)}\n`);
  }

  ref(): void {
    this.refs += 1;
  }

  unref(): void {
    this.unrefs += 1;
  }

  kill(): boolean {
    this.killed = true;
    this.exitCode = 143;
    queueMicrotask(() => this.emit('close', 143));
    return true;
  }

  crash(code = 1): void {
    this.exitCode = code;
    this.emit('close', code);
  }

  /** Requests only — control lines (cancel, toolResult) filtered out. */
  get requests(): Array<Record<string, unknown>> {
    return this.written.filter((m) => m.op !== 'cancel' && m.op !== 'toolResult');
  }

  controls(op: string): Array<Record<string, unknown>> {
    return this.written.filter((m) => m.op === op);
  }
}

export function fakeServer(
  onLine?: (message: Record<string, unknown>, child: FakeChild) => void,
  timeoutMs = 500,
): { server: HelperServer; children: FakeChild[] } {
  const children: FakeChild[] = [];
  const server = new HelperServer('fake', {
    timeoutMs,
    spawner: () => {
      const child = new FakeChild(onLine);
      children.push(child);
      return child as unknown as ChildProcess;
    },
  });
  return { server, children };
}

export const tick = (ms = 5): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
