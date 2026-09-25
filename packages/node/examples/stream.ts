/**
 * Streaming, cancellation and timeouts.
 *
 *   npx tsx examples/stream.ts
 */
import { AbortError, AppleLLM, TimeoutError } from 'apple-llm';

const llm = new AppleLLM({ tier: 'device' });

// Iterate the deltas…
for await (const delta of llm.stream('Write a haiku about bridges.')) process.stdout.write(delta);
process.stdout.write('\n');

// …or await the whole reply, and read usage off the result.
const stream = llm.stream('Name three rivers.');
console.log(await stream);
console.log((await stream.result).usage);

// Cancelling stops the model, not just the promise.
const controller = new AbortController();
setTimeout(() => controller.abort(), 300);
try {
  await llm.text('Write a long essay about bridges.', { signal: controller.signal, maxTokens: 1500 });
} catch (error) {
  console.log(error instanceof AbortError ? 'aborted' : error);
}

try {
  await llm.text('Write a long essay about bridges.', { timeoutMs: 500, maxTokens: 1500 });
} catch (error) {
  console.log(error instanceof TimeoutError ? 'timed out' : error);
}
