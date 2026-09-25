/**
 * Streams that are both `for await`-able and awaitable.
 *
 *   for await (const delta of llm.stream(prompt)) process.stdout.write(delta);
 *   const text = await llm.stream(prompt);          // the whole reply
 *   const { usage } = await llm.stream(prompt).result;
 *
 * The request starts immediately, whichever way it is consumed. Breaking out
 * of a `for await` loop cancels it — nothing keeps generating for a reader who
 * has gone. An error surfaces through whichever of the two you use; a stream
 * nobody consumes never raises an unhandled rejection.
 */

type Item<T> = { value: T } | { done: true } | { error: unknown };

/** An async queue that one reader drains. */
class Channel<T> {
  private readonly items: Array<Item<T>> = [];
  private waiter: ((item: Item<T>) => void) | undefined;
  private closed = false;

  push(value: T): void {
    this.offer({ value });
  }

  end(): void {
    this.offer({ done: true });
  }

  fail(error: unknown): void {
    this.offer({ error });
  }

  private offer(item: Item<T>): void {
    if (this.closed) return;
    if (!('value' in item)) this.closed = true;
    const waiter = this.waiter;
    if (waiter !== undefined) {
      this.waiter = undefined;
      waiter(item);
    } else {
      this.items.push(item);
    }
  }

  next(): Promise<Item<T>> {
    const item = this.items.shift();
    if (item !== undefined) return Promise.resolve(item);
    return new Promise((resolve) => {
      this.waiter = resolve;
    });
  }
}

/**
 * The shared shape: `Chunk`s arrive through the iterator, the promise
 * resolves with `Final`, and `result` carries the full result object.
 */
export class ResultStream<Chunk, Final, Result> implements AsyncIterable<Chunk>, PromiseLike<Final> {
  /** Everything about the finished generation: text, usage, tool calls, finish reason. */
  readonly result: Promise<Result>;
  private readonly channel = new Channel<Chunk>();
  private readonly controller = new AbortController();
  private readonly final: Promise<Final>;
  private iterated = false;
  private finished = false;

  constructor(
    run: (emit: (chunk: Chunk) => void, signal: AbortSignal) => Promise<Result>,
    pick: (result: Result) => Final,
  ) {
    this.result = run((chunk) => this.channel.push(chunk), this.controller.signal).then(
      (result) => {
        this.finished = true;
        this.channel.end();
        return result;
      },
      (error: unknown) => {
        this.finished = true;
        this.channel.fail(error);
        throw error;
      },
    );
    this.final = this.result.then(pick);
    // Consumption is optional: whoever does consume sees the error; nobody
    // else should get an unhandled rejection for it.
    this.result.catch(() => undefined);
    this.final.catch(() => undefined);
  }

  /** Stop generating. The promise and the iterator both reject with an AbortError. */
  abort(reason?: unknown): void {
    this.controller.abort(reason);
  }

  then<A = Final, B = never>(
    onfulfilled?: ((value: Final) => A | PromiseLike<A>) | null,
    onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null,
  ): Promise<A | B> {
    return this.final.then(onfulfilled, onrejected);
  }

  catch<B = never>(onrejected?: ((reason: unknown) => B | PromiseLike<B>) | null): Promise<Final | B> {
    return this.final.catch(onrejected);
  }

  finally(onfinally?: (() => void) | null): Promise<Final> {
    return this.final.finally(onfinally);
  }

  [Symbol.asyncIterator](): AsyncIterator<Chunk> {
    if (this.iterated) throw new Error('A stream can only be iterated once.');
    this.iterated = true;
    return {
      next: async (): Promise<IteratorResult<Chunk>> => {
        const item = await this.channel.next();
        if ('value' in item) return { value: item.value, done: false };
        if ('error' in item) throw item.error;
        return { value: undefined, done: true };
      },
      return: async (): Promise<IteratorResult<Chunk>> => {
        // The reader left early (break, return, a throw in the loop body).
        if (!this.finished) this.abort();
        return { value: undefined, done: true };
      },
    };
  }
}
