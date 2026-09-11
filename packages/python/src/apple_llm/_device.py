"""The on-device tier: Apple's ``FoundationModels`` framework, reached through a
self-compiled Swift helper. Nothing leaves the machine."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from typing import Any, Literal, Optional, Union

from ._compile import ProgressCallback, ensure_binary
from ._errors import (
    AppleLLMError,
    ContextLengthError,
    ModelUnavailableError,
    QuotaError,
    RefusalError,
    SchemaRejectedError,
    to_unavailable_reason,
    unavailable_message,
)
from ._protocol import HelperServer
from ._schema import JsonSchema, to_apple_schema
from ._target import is_apple_silicon_mac, macos_major

#: Sampling temperature. Not zero on purpose.
#:
#: Constrained decoding already guarantees the schema, so greedy decoding buys
#: nothing and reliably degenerates: at 0 the model padded an unbounded array
#: forever, then ran away *inside a single string*, emitting 2.7KB of
#: "tasks-tasks-tasks-...". A 2s call became 20s. Apple honours ``maxItems`` but
#: ignores ``maxLength``, so bounding strings is not available as a fix -- a
#: little sampling is. Nothing about this is a quality/creativity tradeoff.
DEFAULT_TEMPERATURE = 0.4

#: Cap on generated tokens. Left uncapped, the model occasionally runs away and
#: only stops when it exhausts the context window -- one observed run burned
#: ~206s before failing.
DEFAULT_MAX_TOKENS = 2048

_CALL_TIMEOUT = 120.0

#: Apple ships a use case specialised for tagging and topic extraction.
UseCase = Literal["general", "contentTagging"]

#: ``permissive`` selects ``permissiveContentTransformations``, which relaxes the
#: guardrails for content *transformation* -- rewriting or summarising text the
#: default guardrails would refuse to touch.
Guardrails = Literal["default", "permissive"]

#: An image attachment: a bare path, or a path with a Siri-style label so
#: follow-up turns can refer to it ("the receipt").
ImageAttachment = Union[str, dict]

#: Apple's built-in on-device tools (all macOS 27+, all local).
BuiltInTool = Literal["ocr", "barcode", "spotlight"]

_BUILT_IN_TOOLS = frozenset({"ocr", "barcode", "spotlight"})


def assert_tools(tools: Optional[list] = None) -> None:
    """Fail fast on an unknown tool, before spending a model call."""
    if not tools:
        return
    for tool in tools:
        if tool not in _BUILT_IN_TOOLS:
            raise AppleLLMError(
                f'Unknown tool "{tool}" (want {", ".join(sorted(_BUILT_IN_TOOLS))}).',
                "device",
            )


def parse_image_flag(value: str) -> Union[str, dict]:
    """Parse a CLI --image value: "path" or "path::label"."""
    sep = value.rfind("::")
    if sep > 0:
        path, label = value[:sep], value[sep + 2 :].strip()
        if path and label:
            return {"path": path, "label": label}
    return value


def with_documents(prompt: str, documents: Optional[list[str]] = None) -> str:
    """Inline text documents into the prompt client-side.

    The helper's vision path handles images; plain-text sources are cheaper to
    splice here. Binary files are refused loudly -- silently skipping a document
    the caller asked about would be the vision-drop trap by another name.
    """
    if not documents:
        return prompt
    parts = [prompt]
    for doc in documents:
        try:
            size = os.path.getsize(doc)
        except OSError:
            raise AppleLLMError(f"Document not found: {doc}", "device") from None
        if size > 512_000:
            raise AppleLLMError(f"Document too large to inline (>512000 bytes): {doc}", "device")
        try:
            with open(doc, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            raise AppleLLMError(f"Document is not readable text: {doc}", "device") from None
        if "�" in text:
            raise AppleLLMError(f"Document is not readable text: {doc}", "device")
        parts.append(f"\n\n--- Document: {doc} ---\n{text}")
    return "".join(parts)


def sampling_greedy() -> dict[str, Any]:
    """Deterministic, but degenerates under guided generation -- this is the
    ``temperature=0`` trap by another name. Prefer :func:`sampling_seeded`."""
    return {"mode": "greedy"}


def sampling_seeded(seed: int, top_k: int = 50) -> dict[str, Any]:
    """Reproducible sampling *without* greedy's degeneration.

    Returned byte-identical output across three fresh sessions here. Determinism
    relies on a fresh session per request, which is the default.
    """
    return {"mode": "topK", "k": top_k, "seed": seed}


def sampling_threshold(p: float = 0.9, seed: Optional[int] = None) -> dict[str, Any]:
    """Nucleus sampling; pass ``seed`` to make it reproducible."""
    out: dict[str, Any] = {"mode": "threshold", "p": p}
    if seed is not None:
        out["seed"] = seed
    return out


def probe_device(on_progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    """Probe the on-device model. Never raises."""
    if platform.system() != "Darwin":
        return {
            "available": False,
            "reason": f"unsupportedPlatform: this machine reports \"{platform.system()}\"",
        }
    if not is_apple_silicon_mac():
        return {
            "available": False,
            "reason": "deviceNotEligible: Apple Intelligence needs Apple Silicon",
        }
    major = macos_major()
    # Checked before compiling: on macOS 25 the compile fails with an opaque SDK
    # error, and probe() must return a reason rather than burn a build attempt.
    if major is not None and major < 26:
        return {
            "available": False,
            "reason": f"unsupportedOSVersion: needs macOS 26 or later, this is {major}",
        }

    try:
        binary = ensure_binary(on_progress)
    except AppleLLMError as err:
        return {"available": False, "reason": str(err)}

    try:
        result = subprocess.run(  # noqa: S603 - path comes from our own cache
            [binary, "--probe"], capture_output=True, text=True, timeout=30, check=True
        )
        return json.loads(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        return {"available": False, "reason": f"the helper could not be run: {str(err)[:300]}"}


def _raise_for(response: dict[str, Any], raw: str) -> None:
    """Turn a helper failure envelope into the matching typed error."""
    detail = response.get("error") or raw[:200]
    kind = response.get("kind")
    if kind == "availability":
        raise ModelUnavailableError(
            unavailable_message(detail), to_unavailable_reason(detail), "device"
        )
    if kind == "schema":
        raise SchemaRejectedError(
            f"Apple rejected the response schema: {detail}\n"
            "Every object needs a title, an x-order and additionalProperties; enums must "
            "be anyOf/const; unions cannot include null.",
            "device",
        )
    if kind == "context":
        raise ContextLengthError(
            f"The prompt exceeded the on-device context window: {detail}", "device",
            response.get("contextSize") if isinstance(response.get("contextSize"), int) else None,
        )
    if kind == "quota":
        reset = response.get("resetDate")
        parsed: Optional[datetime] = None
        if isinstance(reset, str):
            try:
                parsed = datetime.fromisoformat(reset.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        raise QuotaError(f"Apple rate limited the on-device model: {detail}", "device", parsed)
    if kind == "guardrail":
        raise RefusalError(f"The on-device model declined to answer: {detail}", "device")
    raise AppleLLMError(f"Apple on-device generation failed: {detail}", "device")


class DeviceClient:
    def __init__(
        self,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        self._binary: Optional[str] = None
        self._probe: Optional[dict[str, Any]] = None
        self._server: Optional[HelperServer] = None
        self._use_case = use_case
        self._guardrails = guardrails

    @property
    def label(self) -> str:
        variant = (self._probe or {}).get("variant")
        return f"apple on-device ({variant})" if variant else "apple on-device"

    @property
    def context_size(self) -> Optional[int]:
        return (self._probe or {}).get("contextSize")

    def get_probe(self) -> Optional[dict[str, Any]]:
        return self._probe

    def ensure_ready(self, on_progress: Optional[ProgressCallback] = None) -> None:
        if self._probe is not None and self._probe.get("available"):
            return
        probe = probe_device(on_progress)
        self._probe = probe
        if not probe.get("available"):
            reason = probe.get("reason")
            raise ModelUnavailableError(
                unavailable_message(reason), to_unavailable_reason(reason), "device"
            )
        self._binary = ensure_binary(on_progress)
        if on_progress:
            variant = probe.get("variant")
            extra = f" ({variant}, {probe.get('contextSize')} token context)" if variant else ""
            on_progress(f"apple on-device model ready{extra}")

    def _exchange(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Send one envelope and return the parsed reply, raising on failure."""
        self.ensure_ready()
        binary = self._binary or ensure_binary()
        if self._server is None:
            self._server = HelperServer(binary, timeout=_CALL_TIMEOUT)
        raw = self._server.send(json.dumps(envelope))
        try:
            response = json.loads(raw)
        except ValueError as err:
            raise AppleLLMError(
                f"Apple helper returned an unreadable envelope: {raw[:200]}", "device"
            ) from err
        if response.get("ok") is not True:
            _raise_for(response, raw)
        return response

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        schema: Optional[JsonSchema] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_schema_in_prompt: Optional[bool] = None,
        reuse_session: bool = False,
        session_id: Optional[str] = None,
        tools: Optional[list] = None,
        images: Optional[list] = None,
        documents: Optional[list[str]] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> str:
        """One request. Returns the helper's ``content`` string, unparsed."""
        assert_tools(tools)
        prompt = with_documents(prompt, documents)
        response = self._exchange(
            {
                "instructions": system or "",
                "prompt": prompt,
                "schema": schema,
                "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
                "maxTokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
                "includeSchemaInPrompt": include_schema_in_prompt,
                "reuseSession": reuse_session,
                "sessionId": session_id,
                "tools": tools,
                "images": images,
                "useCase": use_case if use_case is not None else self._use_case,
                "guardrails": guardrails if guardrails is not None else self._guardrails,
                "sampling": sampling,
            }
        )
        if not isinstance(response.get("content"), str):
            raise AppleLLMError("Apple helper returned no content.", "device")
        return response["content"]

    def stream(
        self,
        prompt: str,
        *,
        on_delta=None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reuse_session: bool = False,
        session_id: Optional[str] = None,
        tools: Optional[list] = None,
        images: Optional[list] = None,
        documents: Optional[list[str]] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> str:
        """Streaming text.

        Deltas arrive via ``on_delta`` as the model generates; the return value
        is the full text. Text only: the helper rejects schema+stream, because
        partial JSON is not a usable delta.
        """
        from ._compile import ensure_binary as _ensure

        assert_tools(tools)
        prompt = with_documents(prompt, documents)
        self.ensure_ready()
        binary = self._binary or _ensure()
        if self._server is None:
            self._server = HelperServer(binary, timeout=_CALL_TIMEOUT)
        envelope = {
            "op": "stream",
            "instructions": system or "",
            "prompt": prompt,
            "schema": None,
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
            "maxTokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            "reuseSession": reuse_session,
            "sessionId": session_id,
            "tools": tools,
            "images": images,
            "useCase": use_case if use_case is not None else self._use_case,
            "guardrails": guardrails if guardrails is not None else self._guardrails,
            "sampling": sampling,
        }
        raw = self._server.stream(json.dumps(envelope), on_delta)
        try:
            response = json.loads(raw)
        except ValueError as err:
            raise AppleLLMError(
                f"Apple helper returned an unreadable envelope: {raw[:200]}", "device"
            ) from err
        if response.get("ok") is not True:
            _raise_for(response, raw)
        if not isinstance(response.get("content"), str):
            raise AppleLLMError("Apple helper returned no content.", "device")
        return response["content"]

    def history(self, session_id: str) -> dict[str, Any]:
        """Conversation history for a named session, oldest first."""
        response = self._exchange({"op": "history", "sessionId": session_id, "instructions": ""})
        history = response.get("history")
        return {
            "instructions": response.get("instructions") or "",
            "history": history if isinstance(history, list) else [],
        }

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Drop a named session (and its persisted history), or all of them."""
        self._exchange({"op": "reset", "sessionId": session_id or "", "instructions": ""})

    def count_tokens(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[list] = None,
        tools: Optional[list] = None,
    ) -> dict[str, int]:
        """How many tokens a prompt costs, before sending it.

        The point is to turn a ContextLengthError into an arithmetic check:
        compare against ``contextSize`` and trim, rather than discovering the
        ceiling by hitting it. Counts the instructions too, since they share the
        window.
        """
        response = self._exchange(
            {"op": "countTokens", "instructions": system or "", "prompt": prompt,
             "images": images, "tools": tools}
        )
        tokens = response.get("tokens")
        context_size = response.get("contextSize")
        if not isinstance(tokens, int) or not isinstance(context_size, int):
            raise AppleLLMError(
                f"Apple helper returned an unreadable token count: {tokens!r}/{context_size!r}",
                "device",
            )
        return {
            "tokens": tokens,
            "context_size": context_size,
        }

    def prewarm(self, system: Optional[str] = None) -> None:
        """Load the model assets now so the first real call does not pay for it.

    Cheap and idempotent, but do not expect much on a warm machine: with the
    assets already resident this measured 0.31s against 0.36s for an unprewarmed
    first call -- inside the noise. The win is on a genuinely cold system, where
    the very first call to the framework on this machine took 7.8s. Worth calling
    at startup when you know a request is coming; not worth building around.
        """
        self._exchange({"op": "prewarm", "instructions": system or ""})

    def text(self, prompt: str, **kwargs: Any) -> str:
        kwargs.pop("schema", None)
        return self.complete(prompt, schema=None, **kwargs)

    def json(self, prompt: str, *, schema: JsonSchema, **kwargs: Any) -> Any:
        content = self.complete(prompt, schema=to_apple_schema(schema), **kwargs)
        try:
            return json.loads(content)
        except ValueError as err:
            # Constrained decoding should make this unreachable.
            raise AppleLLMError(
                f"Apple on-device model returned invalid JSON: {content[:200]}", "device"
            ) from err

    def close(self) -> None:
        """Shut the helper process down. Safe to call more than once."""
        if self._server is not None:
            self._server.stop()
            self._server = None
