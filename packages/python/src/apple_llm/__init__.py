"""apple-llm -- one library for Apple's on-device and Private Cloud Compute models.

macOS 26+ on Apple Silicon only. No API key, no account, no developer program.
Extracted from api-scribe (MIT), which discovered and shipped both routes.

    from apple_llm import AppleLLM, probe

    probe()

    with AppleLLM(tier="device") as llm:
        llm.text("Summarize this", system="You are terse.")
        llm.json("Extract the fields", schema=MyPydanticModel)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

from ._cloud import (
    CLOUD_CONTEXT_TOKENS,
    CLOUD_SHORTCUT_NAME,
    CLOUD_SHORTCUT_NAME_WEB,
    CloudClient,
    cloud_setup_hint,
    install_cloud_shortcut,
    probe_cloud,
    shortcut_definition,
)
from ._compile import ProgressCallback, cache_dir, ensure_binary, fingerprint, helper_source
from ._device import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    BuiltInTool,
    DeviceClient,
    Guardrails,
    ImageAttachment,
    UseCase,
    assert_tools,
    parse_image_flag,
    probe_device,
    sampling_greedy,
    sampling_seeded,
    sampling_threshold,
    with_documents,
)
from ._errors import (
    AppleLLMError,
    ContextLengthError,
    ModelUnavailableError,
    QuotaError,
    RefusalError,
    SchemaRejectedError,
    SetupRequiredError,
    TimeoutError,
)
from ._json_recovery import extract_json_span, parse_llm_json, strip_code_fences
from ._schema import JsonSchema, to_apple_schema
from ._target import host_target, is_apple_silicon_mac, target_triple_from

__version__ = "0.1.0"

__all__ = [
    "AppleLLM",
    "AsyncAppleLLM",
    "Conversation",
    "AsyncConversation",
    "probe",
    "capture_screenshot",
    "DeviceClient",
    "CloudClient",
    "probe_device",
    "probe_cloud",
    "install_cloud_shortcut",
    "shortcut_definition",
    "cloud_setup_hint",
    "to_apple_schema",
    "parse_llm_json",
    "strip_code_fences",
    "extract_json_span",
    "is_apple_silicon_mac",
    "target_triple_from",
    "host_target",
    "cache_dir",
    "fingerprint",
    "helper_source",
    "ensure_binary",
    "assert_tools",
    "parse_image_flag",
    "with_documents",
    "AppleLLMError",
    "ModelUnavailableError",
    "SchemaRejectedError",
    "ContextLengthError",
    "QuotaError",
    "TimeoutError",
    "SetupRequiredError",
    "RefusalError",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "UseCase",
    "Guardrails",
    "ImageAttachment",
    "BuiltInTool",
    "sampling_greedy",
    "sampling_seeded",
    "sampling_threshold",
    "CLOUD_SHORTCUT_NAME",
    "CLOUD_SHORTCUT_NAME_WEB",
    "CLOUD_CONTEXT_TOKENS",
    "JsonSchema",
    "ProgressCallback",
    "__version__",
]


def probe(on_progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    """What this machine can actually do.

    Never raises: on Linux, an Intel Mac or macOS 25 it returns
    ``available: False`` with a reason naming the fix.

    ``device`` carries the model's capabilities (vision / guidedGeneration /
    reasoning / toolCalling) on macOS 27+, and ``cloud["quota"]`` carries the
    real Private Cloud Compute quota -- readable from the on-device helper even
    though PCC *inference* needs an entitlement no package can ship.
    """
    device = probe_device(on_progress)
    cloud = probe_cloud()
    if device.get("cloud") is not None:
        cloud["quota"] = device["cloud"]
    return {"device": device, "cloud": cloud}


def _schema_of(schema: Any) -> tuple[JsonSchema, Any]:
    """Accept a plain dict or a pydantic model class.

    Returns ``(json_schema, model_or_None)``. Pydantic is an optional extra
    (``apple-llm[pydantic]``); plain dict schemas work with no extras, so this
    never imports it unless the caller passes a model.
    """
    if isinstance(schema, dict):
        return schema, None
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema(), schema
    raise TypeError(
        "schema must be a dict or a pydantic BaseModel subclass, "
        f"not {type(schema).__name__}"
    )


class AppleLLM:
    """The main entry point.

    ``tier`` is ``"device"``, ``"cloud"`` or ``"auto"``. ``auto`` means on-device
    when available, else cloud when the shortcut is installed, else an error
    naming the setup step.
    """

    def __init__(
        self,
        tier: str = "auto",
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> None:
        if tier not in ("device", "cloud", "auto"):
            raise ValueError(f'tier must be "device", "cloud" or "auto", not {tier!r}')
        self.tier = tier
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._on_progress = on_progress
        self._use_case = use_case
        self._guardrails = guardrails
        self._sampling = sampling
        self._device: Optional[DeviceClient] = None
        self._cloud: Optional[CloudClient] = None
        self._resolved: Optional[str] = None

    @property
    def label(self) -> str:
        """Human-readable name of the tier in use, for logs."""
        if self._resolved == "cloud":
            return self._cloud.label if self._cloud else "apple private cloud compute"
        if self._resolved == "device":
            return self._device.label if self._device else "apple on-device"
        return f"apple ({self.tier})"

    def ensure_ready(self, on_progress: Optional[ProgressCallback] = None) -> None:
        """Resolve the tier and do any one-time setup.

        Called automatically, but exposed so a caller can pay the compile cost up
        front with a progress bar.
        """
        progress = on_progress or self._on_progress
        if self._resolved is not None:
            return

        if self.tier == "device":
            self._device = self._device or DeviceClient(self._use_case, self._guardrails)
            self._device.ensure_ready(progress)
            self._resolved = "device"
            return
        if self.tier == "cloud":
            self._cloud = self._cloud or CloudClient()
            self._cloud.ensure_ready(progress)
            # Best effort: the device helper can read the PCC quota, which turns
            # an exhausted quota into an immediate error, not a wasted round trip.
            try:
                self._cloud.set_quota(probe_device().get("cloud"))
            except Exception:  # noqa: BLE001 - a quota hint is never worth failing over
                pass
            self._resolved = "cloud"
            return

        self._device = self._device or DeviceClient(self._use_case, self._guardrails)
        try:
            self._device.ensure_ready(progress)
            self._resolved = "device"
            return
        except AppleLLMError as device_error:
            self._cloud = self._cloud or CloudClient()
            try:
                self._cloud.ensure_ready(progress)
                # Best effort: same quota fail-fast as the explicit cloud path.
                try:
                    self._cloud.set_quota(probe_device().get("cloud"))
                except Exception:  # noqa: BLE001 - a quota hint is never worth failing over
                    pass
                self._resolved = "cloud"
                return
            except AppleLLMError as cloud_error:
                reason = getattr(device_error, "reason", "unknown")
                raise ModelUnavailableError(
                    "No Apple model is available.\n\n"
                    f"On-device: {device_error}\n\nCloud: {cloud_error}",
                    reason,
                ) from cloud_error

    def _device_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Per-call device settings, falling back to the constructor defaults."""
        return {
            "temperature": overrides.get("temperature") if overrides.get("temperature") is not None else self._temperature,
            "max_tokens": overrides.get("max_tokens") if overrides.get("max_tokens") is not None else self._max_tokens,
            "images": overrides.get("images"),
            "documents": overrides.get("documents"),
            "session_id": overrides.get("session_id"),
            "tools": overrides.get("tools"),
            "use_case": overrides.get("use_case") if overrides.get("use_case") is not None else self._use_case,
            "guardrails": overrides.get("guardrails") if overrides.get("guardrails") is not None else self._guardrails,
            "sampling": overrides.get("sampling") if overrides.get("sampling") is not None else self._sampling,
        }

    def _assert_device_only(self, *, images: Any = None, session_id: Any = None,
                            tools: Any = None, documents: Any = None) -> None:
        if self._resolved == "cloud":
            if images:
                raise AppleLLMError("images needs the on-device tier.", "cloud")
            if session_id is not None:
                raise AppleLLMError("session_id needs the on-device tier.", "cloud")
            if tools:
                raise AppleLLMError("tools needs the on-device tier.", "cloud")
            if documents:
                raise AppleLLMError("documents needs the on-device tier.", "cloud")

    def text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        images: Optional[list] = None,
        documents: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        tools: Optional[list] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> str:
        self.ensure_ready()
        if self._resolved == "cloud":
            self._assert_device_only(images=images, session_id=session_id,
                                     tools=tools, documents=documents)
            return self._cloud.text(prompt, system=system, web_search=web_search)  # type: ignore[union-attr]
        return self._device.text(  # type: ignore[union-attr]
            prompt,
            system=system,
            **self._device_kwargs(
                temperature=temperature, max_tokens=max_tokens, images=images,
                documents=documents, session_id=session_id, tools=tools,
                use_case=use_case, guardrails=guardrails, sampling=sampling,
            ),
        )

    def stream(
        self,
        prompt: str,
        *,
        on_delta=None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list] = None,
        documents: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        tools: Optional[list] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> str:
        """Streaming text (device tier only).

        Deltas arrive via ``on_delta`` as the model generates; the return value
        is the full text.
        """
        self.ensure_ready()
        if self._resolved != "device":
            raise AppleLLMError("stream needs the on-device tier.", "cloud")
        return self._device.stream(  # type: ignore[union-attr]
            prompt,
            on_delta=on_delta,
            system=system,
            **self._device_kwargs(
                temperature=temperature, max_tokens=max_tokens, images=images,
                documents=documents, session_id=session_id, tools=tools,
                use_case=use_case, guardrails=guardrails, sampling=sampling,
            ),
        )

    def history(self, session_id: str) -> dict[str, Any]:
        """Conversation history for a named session (device tier only)."""
        self.ensure_ready()
        if self._resolved != "device":
            raise AppleLLMError("history needs the on-device tier.", "cloud")
        return self._device.history(session_id)  # type: ignore[union-attr]

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Drop a named session, or all sessions when omitted."""
        self.ensure_ready()
        if self._resolved == "device":
            self._device.reset_session(session_id)  # type: ignore[union-attr]

    def conversation(self, session_id: str, system: Optional[str] = None) -> "Conversation":
        """A named conversation sharing one native transcript, like one thread
        in the Siri app."""
        return Conversation(self, session_id, system)

    def rewrite(self, text: str, instruction: Optional[str] = None, **kwargs: Any) -> str:
        """Rewrite text (device tier; uses permissive guardrails)."""
        how = instruction or "Keep the meaning, improve clarity."
        kwargs.setdefault("guardrails", "permissive")
        kwargs.setdefault("system", "You rewrite text. Reply with only the rewritten text, no commentary.")
        return self.text(f"Rewrite the following text. {how}\n\n---\n{text}", **kwargs)

    def proofread(self, text: str, **kwargs: Any) -> str:
        """Fix spelling, grammar and punctuation (device tier)."""
        kwargs.setdefault("guardrails", "permissive")
        kwargs.setdefault("system", "You proofread text. Reply with only the corrected text, no commentary.")
        return self.text(
            "Fix spelling, grammar and punctuation in the following text. "
            f"Preserve the meaning and tone.\n\n---\n{text}", **kwargs)

    def summarize(self, text: str, length: str = "one short paragraph", **kwargs: Any) -> str:
        """Summarize text (device tier)."""
        kwargs.setdefault("guardrails", "permissive")
        kwargs.setdefault("system", "You summarize text tersely.")
        return self.text(f"Summarize the following text in {length}.\n\n---\n{text}", **kwargs)

    def draft(self, topic: str, kind: str = "short draft", **kwargs: Any) -> str:
        """Draft from scratch (device tier)."""
        kwargs.setdefault("system", "You are a helpful writing assistant.")
        return self.text(f"Write a {kind} about the following topic.\n\n---\n{topic}", **kwargs)

    def tone(self, text: str, tone: str, **kwargs: Any) -> str:
        """Rewrite text in another tone (device tier)."""
        kwargs.setdefault("guardrails", "permissive")
        kwargs.setdefault("system", "You rewrite text. Reply with only the rewritten text, no commentary.")
        return self.text(f"Rewrite the following text to sound more {tone}.\n\n---\n{text}", **kwargs)

    def ask_screen(self, question: str, mode: str = "interactive", **kwargs: Any) -> str:
        """Ask about what's on screen (device tier, macOS 27+).

        Captures a screenshot -- interactive selection by default, like
        Cmd+Shift+Space Visual Intelligence -- and asks the model about it.
        """
        self.ensure_ready()
        if self._resolved != "device":
            raise AppleLLMError("ask_screen needs the on-device tier.", "cloud")
        shot = capture_screenshot(mode)
        try:
            images = list(kwargs.pop("images", None) or [])
            images.append(shot)
            return self.text(question, images=images, **kwargs)
        finally:
            import os as _os

            try:
                _os.unlink(shot)
            except OSError:
                pass

    def count_tokens(
        self, prompt: str, *, system: Optional[str] = None,
        images: Optional[list] = None, tools: Optional[list] = None,
    ) -> dict[str, int]:
        """How many tokens a prompt costs, before sending it. Device tier only."""
        self.ensure_ready()
        if self._resolved != "device":
            raise AppleLLMError("count_tokens needs the on-device tier.", "cloud")
        return self._device.count_tokens(prompt, system=system, images=images, tools=tools)  # type: ignore[union-attr]

    def prewarm(self, system: Optional[str] = None) -> None:
        """Load the model assets now so the first real call does not pay for it.

        See :meth:`DeviceClient.prewarm` for what it is actually worth (little,
        on a machine where the assets are already resident).
        """
        self.ensure_ready()
        if self._resolved == "device":
            self._device.prewarm(system)  # type: ignore[union-attr]

    def json(
        self,
        prompt: str,
        *,
        schema: Any,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        web_search: bool = False,
        images: Optional[list] = None,
        documents: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        tools: Optional[list] = None,
        use_case: Optional[UseCase] = None,
        guardrails: Optional[Guardrails] = None,
        sampling: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Ask for JSON.

        On device the schema is *guaranteed* by constrained decoding. On cloud it
        is requested in the prompt and recovered from the reply -- the cloud tier
        has no constrained decoding, so a bad shape is possible there.

        ``schema`` may be a plain dict or a pydantic ``BaseModel`` subclass; a
        model gets a validated instance back rather than a dict.
        """
        json_schema, model = _schema_of(schema)
        self.ensure_ready()
        if self._resolved == "cloud":
            self._assert_device_only(images=images, session_id=session_id,
                                     tools=tools, documents=documents)
            result = self._cloud.json(  # type: ignore[union-attr]
                prompt, schema=json_schema, system=system, web_search=web_search
            )
        else:
            result = self._device.json(  # type: ignore[union-attr]
                prompt,
                schema=json_schema,
                system=system,
                **self._device_kwargs(
                    temperature=temperature, max_tokens=max_tokens, images=images,
                    documents=documents, session_id=session_id, tools=tools,
                    use_case=use_case, guardrails=guardrails, sampling=sampling,
                ),
            )
        return model.model_validate(result) if model is not None else result

    def complete_json(self, system: str, user: str, schema: Any) -> Any:
        """api-scribe's ``LlmClient`` shape, so it can drop its four files and
        depend on this instead. Not used internally."""
        return self.json(user, schema=schema, system=system)

    def close(self) -> None:
        """Release the long-lived helper process. Safe to call more than once."""
        if self._device is not None:
            self._device.close()
        if self._cloud is not None:
            self._cloud.close()

    def __enter__(self) -> "AppleLLM":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncAppleLLM:
    """The same API, awaitable.

    Each call runs the blocking client in a worker thread. The helper process
    serialises requests anyway -- Apple's framework serialises inference, so
    concurrency buys nothing -- which makes a thread the honest shape here and
    avoids a second copy of the protocol that could drift from the sync one.
    """

    def __init__(self, tier: str = "auto", **kwargs: Any) -> None:
        self._inner = AppleLLM(tier, **kwargs)
        self._lock = asyncio.Lock()

    @property
    def tier(self) -> str:
        return self._inner.tier

    @property
    def label(self) -> str:
        return self._inner.label

    async def ensure_ready(self, on_progress: Optional[ProgressCallback] = None) -> None:
        async with self._lock:
            await asyncio.to_thread(self._inner.ensure_ready, on_progress)

    async def text(self, prompt: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.text(prompt, **kwargs))

    async def json(self, prompt: str, **kwargs: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.json(prompt, **kwargs))

    async def stream(self, prompt: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.stream(prompt, **kwargs))

    async def history(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.history(session_id))

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        async with self._lock:
            await asyncio.to_thread(lambda: self._inner.reset_session(session_id))

    def conversation(self, session_id: str, system: Optional[str] = None) -> "AsyncConversation":
        return AsyncConversation(self, session_id, system)

    async def rewrite(self, text: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.rewrite(text, **kwargs))

    async def proofread(self, text: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.proofread(text, **kwargs))

    async def summarize(self, text: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.summarize(text, **kwargs))

    async def draft(self, topic: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.draft(topic, **kwargs))

    async def tone(self, text: str, tone: str, **kwargs: Any) -> str:
        async with self._lock:
            return await asyncio.to_thread(lambda: self._inner.tone(text, tone, **kwargs))

    async def complete_json(self, system: str, user: str, schema: Any) -> Any:
        return await self.json(user, schema=schema, system=system)

    def close(self) -> None:
        self._inner.close()

    async def __aenter__(self) -> "AsyncAppleLLM":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self._inner.close)


class Conversation:
    """One thread in the Siri-app sense: every call shares one sessionId."""

    def __init__(self, llm: AppleLLM, session_id: str, system: Optional[str] = None) -> None:
        self._llm = llm
        self.session_id = session_id
        self._system = system

    def text(self, prompt: str, **kwargs: Any) -> str:
        kwargs.setdefault("system", self._system)
        return self._llm.text(prompt, session_id=self.session_id, **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> str:
        kwargs.setdefault("system", self._system)
        return self._llm.stream(prompt, session_id=self.session_id, **kwargs)

    def history(self) -> dict[str, Any]:
        return self._llm.history(self.session_id)

    def reset(self) -> None:
        self._llm.reset_session(self.session_id)


class AsyncConversation:
    """Async mirror of :class:`Conversation`."""

    def __init__(self, llm: AsyncAppleLLM, session_id: str, system: Optional[str] = None) -> None:
        self._llm = llm
        self.session_id = session_id
        self._system = system

    async def text(self, prompt: str, **kwargs: Any) -> str:
        kwargs.setdefault("system", self._system)
        return await self._llm.text(prompt, session_id=self.session_id, **kwargs)

    async def stream(self, prompt: str, **kwargs: Any) -> str:
        kwargs.setdefault("system", self._system)
        return await self._llm.stream(prompt, session_id=self.session_id, **kwargs)

    async def history(self) -> dict[str, Any]:
        return await self._llm.history(self.session_id)

    async def reset(self) -> None:
        await self._llm.reset_session(self.session_id)


def capture_screenshot(mode: str = "interactive") -> str:
    """Capture a screenshot to a temp file (Visual Intelligence entry point).

    ``mode`` is ``interactive`` (drag to select, like Cmd+Shift+Space),
    ``window`` or ``fullscreen``. Returns the PNG path; the caller removes it.
    """
    import subprocess
    import tempfile

    if mode not in ("interactive", "window", "fullscreen"):
        raise ValueError(f'mode must be interactive, window or fullscreen, not {mode!r}')
    args = {"interactive": ["-i", "-x"], "window": ["-w", "-x"], "fullscreen": ["-x"]}[mode]
    handle, path = tempfile.mkstemp(prefix="apple-llm-screen-", suffix=".png")
    try:
        subprocess.run(["screencapture", *args, path], timeout=120, check=True)  # noqa: S603,S607
    except (OSError, subprocess.SubprocessError) as err:
        raise AppleLLMError(f"Could not capture a screenshot (screencapture failed): {err}") from err
    finally:
        try:
            import os as _os

            _os.close(handle)
        except OSError:
            pass
    try:
        import os as _os

        if _os.path.getsize(path) == 0:
            raise AppleLLMError("Screenshot capture was cancelled or produced no image.", "device")
    except OSError:
        raise AppleLLMError("Screenshot capture was cancelled or produced no image.", "device")
    return path
