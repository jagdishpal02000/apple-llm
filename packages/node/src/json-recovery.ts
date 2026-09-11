/**
 * Recover JSON from free text.
 *
 * Only the cloud tier needs this. On device, constrained decoding guarantees the
 * shape and the reply is already JSON. Private Cloud Compute returns prose, so
 * the object has to be dug out of whatever the model wrapped around it.
 *
 * api-scribe's version only stripped fences when the reply *started* with one,
 * which fails on the common "Here is the JSON:\n```json\n{...}\n```". This one
 * also scans for a balanced object or array.
 */

/** Strip a Markdown code fence anywhere in the text, preferring a json-tagged one. */
export function stripCodeFences(text: string): string {
  const trimmed = text.trim();
  const fenced = /```(?:json)?\s*\n?([\s\S]*?)```/i.exec(trimmed);
  if (fenced?.[1] !== undefined) return fenced[1].trim();
  if (trimmed.startsWith('```')) {
    return trimmed.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  }
  return trimmed;
}

/**
 * The first balanced `{...}` or `[...]` in the text, respecting strings and
 * escapes so a brace inside a string value cannot end the scan early.
 */
export function extractJsonSpan(text: string): string | null {
  for (let start = 0; start < text.length; start += 1) {
    const open = text[start];
    if (open !== '{' && open !== '[') continue;
    const close = open === '{' ? '}' : ']';
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i += 1) {
      const ch = text[i];
      if (escaped) {
        escaped = false;
// eslint-disable-next-line no-continue
        continue;
      }
      if (ch === '\\') {
        if (inString) escaped = true;
        continue;
      }
      if (ch === '"') {
        inString = !inString;
        continue;
      }
      if (inString) continue;
      if (ch === open) depth += 1;
      else if (ch === close) {
        depth -= 1;
        if (depth === 0) return text.slice(start, i + 1);
      }
    }
  }
  return null;
}

/**
 * Parse a model reply as JSON, tolerating fences and surrounding prose.
 * Throws with a truncated echo of the reply, which is what a caller needs to see.
 */
export function parseLlmJson(raw: string): unknown {
  const text = stripCodeFences(raw);
  try {
    return JSON.parse(text);
  } catch {
    /* fall through to the span scan */
  }
  // Try every balanced span in order: the first brace run may be prose
  // ("Use {brackets} then ...") rather than the JSON payload. Advance by one
  // past the span's start (not past its end) so a valid object nested inside
  // an invalid outer span is still found.
  let offset = 0;
  while (offset < text.length) {
    const span = extractJsonSpan(text.slice(offset));
    if (span === null) break;
    try {
      return JSON.parse(span);
    } catch {
      /* not this span — advance and try the next */
    }
    const idx = text.indexOf(span, offset);
    offset = (idx === -1 ? offset : idx) + 1;
  }
  throw new Error(`Model returned invalid JSON: ${raw.slice(0, 200)}`);
}
