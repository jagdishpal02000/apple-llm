import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import type { ChildProcess } from 'node:child_process';
import { describe, expect, it, vi } from 'vitest';
import { HelperServer } from '../src/protocol.js';
import { TimeoutError } from '../src/errors.js';

/**
 * A scripted stand-in for the Swift helper, so the newline-framed protocol can
 * be tested on any machine — none of this needs Apple hardware.
 */
class FakeChild extends EventEmitter {
  stdin = new PassThrough();
  stdout = new PassThrough();
  stderr = new PassThrough();
  exitCode: number | null = null;
  killed = false;
  written: string[] = [];

  constructor(private readonly onLine?: (line: string, child: FakeChild) => void) {
    super();
    this.stdin.on('data', (chunk: Buffer) => {
      for (const line of chunk.toString().split('\n')) {
        if (line.trim() === '') continue;
        this.written.push(line);
        this.onLine?.(line, this);
      }
    });
  }

  reply(text: string): void {
    this.stdout.write(`${text}\n`);
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
}

function serverWith(onLine?: (line: string, child: FakeChild) => void): {
  server: HelperServer;
  children: FakeChild[];
} {
  const children: FakeChild[] = [];
  const server = new HelperServer('fake', {
    timeoutMs: 200,
    spawner: () => {
      const child = new FakeChild(onLine);
      children.push(child);
      return child as unknown as ChildProcess;
    },
  });
  return { server, children };
}

describe('HelperServer', () => {
  it('matches responses to requests in order', async () => {
    // Reply out of band, one line per request, so ordering is the only thing
    // pairing a reply with its caller.
    const { server, children } = serverWith((line, child) => {
      const n = (JSON.parse(line) as { n: number }).n;
      setTimeout(() => child.reply(JSON.stringify({ ok: true, content: `reply-${n}` })), 10 * (3 - n));
    });

    const results = await Promise.all([1, 2, 3].map((n) => server.send(JSON.stringify({ n }))));
    expect(results.map((r) => (JSON.parse(r) as { content: string }).content)).toEqual([
      'reply-1',
      'reply-2',
      'reply-3',
    ]);
    // Serialised: the requests were written one at a time, in order.
    expect(children).toHaveLength(1);
    expect(children[0].written.map((l) => (JSON.parse(l) as { n: number }).n)).toEqual([1, 2, 3]);
    server.stop();
  });

  it('reassembles a response split across chunks', async () => {
    const { server } = serverWith((_line, child) => {
      child.stdout.write('{"ok":true,"con');
      setTimeout(() => child.stdout.write('tent":"whole"}\n'), 5);
    });
    const raw = await server.send('{}');
    expect((JSON.parse(raw) as { content: string }).content).toBe('whole');
    server.stop();
  });

  it('skips blank lines and trailing partials in a chunk', async () => {
    // Only one request is ever outstanding — `send` chains — so two *paired*
    // responses cannot arrive together. What can share a chunk is a blank line,
    // or the start of a line the helper has not finished writing.
    const { server } = serverWith((_line, child) => {
      child.stdout.write('\n\n{"ok":true,"content":"a"}\n{"ok":tr');
    });
    const raw = await server.send('{"n":1}');
    expect((JSON.parse(raw) as { content: string }).content).toBe('a');
    server.stop();
  });

  it('serialises: the next request is not written until the last one replies', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const { server } = serverWith((_line, child) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      setTimeout(() => {
        inFlight -= 1;
        child.reply('{"ok":true,"content":"x"}');
      }, 5);
    });
    await Promise.all([1, 2, 3, 4].map(() => server.send('{}')));
    // Apple serialises inference anyway: 4 concurrent requests measured 29.19s
    // against 29.45s sequentially, so queueing costs nothing and keeps replies
    // unambiguously paired with their requests.
    expect(maxInFlight).toBe(1);
    server.stop();
  });

  it('drops the child on timeout rather than desynchronizing the queue', async () => {
    let replies = 0;
    const { server, children } = serverWith((line, child) => {
      replies += 1;
      // Ignore the first request entirely; answer everything after it.
      if (replies === 1) return;
      child.reply(JSON.stringify({ ok: true, content: 'fresh' }));
    });

    await expect(server.send('{"n":1}')).rejects.toBeInstanceOf(TimeoutError);
    expect(children[0].killed).toBe(true);

    // The late reply must not be handed to the next caller: a new child is
    // started, and the next request gets its own answer.
    const raw = await server.send('{"n":2}');
    expect((JSON.parse(raw) as { content: string }).content).toBe('fresh');
    expect(children).toHaveLength(2);
    server.stop();
  });

  it('rejects everything queued when the child crashes mid-flight', async () => {
    const { server, children } = serverWith();
    const pending = server.send('{"n":1}');
    // Let the write land, then kill the process under it.
    await new Promise((r) => setTimeout(r, 10));
    children[0].crash(9);
    await expect(pending).rejects.toThrow(/exited with code 9/);
    server.stop();
  });

  it('starts a fresh child after a crash', async () => {
    const { server, children } = serverWith((_line, child) => {
      if (children.length === 1) {
        setTimeout(() => child.crash(1), 5);
        return;
      }
      child.reply(JSON.stringify({ ok: true, content: 'recovered' }));
    });
    await expect(server.send('{"n":1}')).rejects.toThrow(/exited with code 1/);
    const raw = await server.send('{"n":2}');
    expect((JSON.parse(raw) as { content: string }).content).toBe('recovered');
    server.stop();
  });

  it('never writes a literal newline into a request line', async () => {
    const { server, children } = serverWith((_line, child) => {
      child.reply('{"ok":true,"content":"ok"}');
    });
    await server.send('{"a":"one\ntwo"}');
    expect(children[0].written).toHaveLength(1);
    expect(children[0].written[0]).not.toContain('\n');
    server.stop();
  });

  it('streams deltas to onDelta and resolves with the final line', async () => {
    const { server } = serverWith((_line, child) => {
      child.reply('{"ok":true,"delta":"Hel","done":false}');
      child.reply('{"ok":true,"delta":"lo","done":false}');
      child.reply('{"ok":true,"content":"Hello","done":true}');
    });
    const deltas: string[] = [];
    const raw = await server.stream('{"op":"stream"}', (d) => deltas.push(d));
    expect(deltas).toEqual(['Hel', 'lo']);
    expect((JSON.parse(raw) as { content: string }).content).toBe('Hello');
    server.stop();
  });

  it('keeps the next request ordered behind a stream', async () => {
    const { server } = serverWith((line, child) => {
      const op = (JSON.parse(line) as { op?: string }).op;
      if (op === 'stream') {
        child.reply('{"ok":true,"delta":"a","done":false}');
        child.reply('{"ok":true,"content":"a","done":true}');
      } else {
        child.reply('{"ok":true,"content":"after"}');
      }
    });
    const deltas: string[] = [];
    const [streamed, after] = await Promise.all([
      server.stream('{"op":"stream"}', (d) => deltas.push(d)),
      server.send('{"op":"generate"}'),
    ]);
    expect((JSON.parse(streamed) as { content: string }).content).toBe('a');
    expect((JSON.parse(after) as { content: string }).content).toBe('after');
    expect(deltas).toEqual(['a']);
    server.stop();
  });

  it('surfaces a stream error envelope to the caller', async () => {
    const { server } = serverWith((_line, child) => {
      child.reply('{"ok":false,"kind":"unsupported","error":"streaming with a schema is not supported"}');
    });
    const raw = await server.stream('{"op":"stream"}', () => undefined);
    expect((JSON.parse(raw) as { ok: boolean }).ok).toBe(false);
    server.stop();
  });
});
