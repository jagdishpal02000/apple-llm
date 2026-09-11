"""Recover JSON from free text.

Only the cloud tier needs this. On device, constrained decoding guarantees the
shape and the reply is already JSON. Private Cloud Compute returns prose, so the
object has to be dug out of whatever the model wrapped around it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Strip a Markdown code fence anywhere in the text, preferring a json one."""
    trimmed = text.strip()
    match = _FENCE.search(trimmed)
    if match is not None:
        return match.group(1).strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```(?:json)?\s*", "", trimmed, flags=re.IGNORECASE)
        return re.sub(r"```\s*$", "", trimmed).strip()
    return trimmed


def extract_json_span(text: str) -> Optional[str]:
    """The first balanced ``{...}`` or ``[...]``, respecting strings and escapes
    so a brace inside a string value cannot end the scan early."""
    for start, open_ch in enumerate(text):
        if open_ch not in "{[":
            continue
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_llm_json(raw: str) -> Any:
    """Parse a model reply as JSON, tolerating fences and surrounding prose."""
    text = strip_code_fences(raw)
    try:
        return json.loads(text)
    except ValueError:
        pass
    pos = 0
    while pos < len(text):
        span = extract_json_span(text[pos:])
        if span is None:
            break
        try:
            return json.loads(span)
        except ValueError:
            pass
        # Advance past this span's start so the next iteration finds the
        # following balanced span (e.g. `Use {brackets} then {"a":1}`).
        nxt = text.find(span, pos)
        if nxt == -1:
            break
        pos = nxt + 1
    raise ValueError(f"Model returned invalid JSON: {raw[:200]}")
