"""Apple's Private Cloud Compute model, reached through the Shortcuts action
``is.workflow.actions.askllm``.

Why this route rather than the framework: macOS 27 exposes
``FoundationModels.PrivateCloudComputeLanguageModel`` as public API and it even
reports ``isAvailable: true``, but every call fails with ``ModelManagerError
1046`` unless the process carries ``com.apple.developer.private-cloud-compute``.
That entitlement is AMFI-restricted -- an ad-hoc-signed binary carrying it is
SIGKILLed (exit 137), and wrapping it in a signed .app with a real bundle ID does
not help. It needs a paid Developer Program provisioning profile, which no
installable package can ship. Shortcuts.app already holds the entitlement and
``/usr/bin/shortcuts`` is public, so a generated shortcut is the only route that
works. Do not spend time re-confirming this.

**This tier sends your prompt off the machine. It is not local.**

Measured on an M4 Air: ~2s typical, ~11s at 14k tokens.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any, Optional

from ._compile import ProgressCallback
from ._errors import (
    AppleLLMError,
    ContextLengthError,
    QuotaError,
    SetupRequiredError,
)
from ._errors import TimeoutError as AppleTimeoutError
from ._json_recovery import parse_llm_json
from ._schema import JsonSchema
from ._target import is_apple_silicon_mac

#: Distinctive, so ``shortcuts run`` cannot match a shortcut the user wrote.
CLOUD_SHORTCUT_NAME = "Apple LLM Cloud"
#: Web search is fixed in the shortcut at install time, so it needs its own copy.
CLOUD_SHORTCUT_NAME_WEB = "Apple LLM Cloud Web"

#: Context window, measured empirically: 14.4k succeeds, ~33.5k is refused.
CLOUD_CONTEXT_TOKENS = 32_768

#: Hard ceiling on one call. Generous next to the ~2s typical case because the
#: failure it guards is not slowness but a *wedged* run: a misconfigured shortcut
#: whose prompt parameter did not bind pops an interactive "Request..." panel and
#: then waits for a human forever.
_CALL_TIMEOUT = 120.0

#: Import is asynchronous and unhurried (5-15s typical).
_IMPORT_TIMEOUT = 90.0
_IMPORT_POLL = 2.0


def shortcut_definition(web_search: bool = False) -> dict[str, Any]:
    """The shortcut definition, as plain JSON.

    Every key here was confirmed against a shortcut built in the Shortcuts GUI
    and exported, rather than guessed -- the difference matters because a wrong
    parameter name does not fail loudly. Shortcuts imports the action, silently
    discards the unrecognised parameter, and the action then blocks on its
    interactive prompt at run time, forever. This is the single most expensive
    mistake available on this path.

    In particular the prompt key is ``WFLLMPrompt``, *not* the ``WFInput`` that
    the action's own localised strings suggest.

    The model key is deliberately absent: with no model key the action uses its
    default, which is the Cloud (Private Cloud Compute) tier.
    """
    parameters: dict[str, Any] = {
        # Stable UUID: re-running setup re-imports the same action rather than
        # accumulating variants.
        "UUID": (
            "B47F1A90-2D5E-4C83-A6F1-9E0C7B34D215"
            if web_search
            else "3827CFFE-3A65-4456-BFA0-49EF061ACEE3"
        ),
        # Plain text out. "Automatic" reshapes the result to suit whatever action
        # comes next, and nothing comes next here.
        "WFGenerativeResultType": "Text",
        # The shortcut's own input, spliced in as a variable. U+FFFC is the
        # object-replacement character marking the attachment's position.
        "WFLLMPrompt": {
            "Value": {
                "string": "￼",
                "attachmentsByRange": {"{0, 1}": {"Type": "ExtensionInput"}},
            },
            "WFSerializationType": "WFTextTokenString",
        },
    }
    # "Use Broad World Knowledge" -- the live-internet flag. Only set when asked,
    # so the default shortcut stays byte-identical to the confirmed-good one.
    if web_search:
        parameters["WFAllowWebSearch"] = True

    return {
        "WFWorkflowMinimumClientVersionString": "900",
        "WFWorkflowMinimumClientVersion": 900,
        "WFWorkflowClientVersion": "5037.0.17",
        "WFWorkflowIcon": {
            "WFWorkflowIconStartColor": 431817727,
            "WFWorkflowIconGlyphNumber": 61440,
        },
        "WFWorkflowOutputContentItemClasses": [],
        "WFWorkflowHasOutputFallback": False,
        "WFWorkflowActions": [
            {
                "WFWorkflowActionIdentifier": "is.workflow.actions.askllm",
                "WFWorkflowActionParameters": parameters,
            }
        ],
        "WFWorkflowInputContentItemClasses": ["WFStringContentItem", "WFRichTextContentItem"],
        "WFWorkflowImportQuestions": [],
        "WFQuickActionSurfaces": [],
        "WFWorkflowTypes": ["WFWorkflowTypeShowInSearch"],
        "WFWorkflowHasShortcutInputVariables": True,
    }


def _list_shortcuts() -> list[str]:
    try:
        result = subprocess.run(
            ["shortcuts", "list"], capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.split("\n") if line.strip()]


def install_count(name: str = CLOUD_SHORTCUT_NAME) -> int:
    """How many installed shortcuts carry the name. >1 breaks ``shortcuts run``."""
    return sum(1 for line in _list_shortcuts() if line == name)


def cloud_setup_hint() -> str:
    return (
        "The cloud tier needs a one-time shortcut installed.\n"
        "Run:  apple-llm setup-cloud\n"
        "It generates and signs the shortcut on this machine -- no account, no API key, "
        "nothing uploaded."
    )


def _duplicate_message(count: int, name: str) -> str:
    return (
        f'{count} shortcuts are named "{name}"; `shortcuts run` cannot tell them apart '
        "and fails with \"Couldn't find shortcut\".\n"
        "Delete the duplicates in the Shortcuts app, then try again."
    )


def install_cloud_shortcut(
    on_progress: Optional[ProgressCallback] = None,
    force: bool = False,
    web_search: bool = False,
) -> None:
    """Generate, sign and install the shortcut. Idempotent unless ``force``.

    Signing needs no developer account and no signing identity: ``shortcuts sign
    -m anyone`` issues a per-signature certificate that chains to Apple Root CA -
    G3 on the device. Every user signs their own copy, so nothing has to be
    pre-signed, hosted, or shipped in the package.
    """
    name = CLOUD_SHORTCUT_NAME_WEB if web_search else CLOUD_SHORTCUT_NAME
    existing = install_count(name)
    if existing > 1:
        raise AppleLLMError(_duplicate_message(existing, name), "cloud")
    if existing == 1 and not force:
        if on_progress:
            on_progress(f'"{name}" is already installed')
        return

    with tempfile.TemporaryDirectory(prefix="apple-llm-shortcut-") as directory:
        # plutil converts JSON to a plist, which saves hand-rolling plist XML.
        # `shortcuts sign` insists on a .shortcut extension for its input.
        json_path = os.path.join(directory, "definition.json")
        unsigned = os.path.join(directory, "unsigned.shortcut")
        signed = os.path.join(directory, f"{name}.shortcut")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(shortcut_definition(web_search), handle)
        subprocess.run(
            ["plutil", "-convert", "binary1", "-o", unsigned, json_path],
            capture_output=True, text=True, timeout=30, check=True,
        )

        if on_progress:
            on_progress("signing the shortcut locally")
        subprocess.run(
            ["shortcuts", "sign", "-m", "anyone", "-i", unsigned, "-o", signed],
            capture_output=True, text=True, timeout=60, check=True,
        )

        # `open` hands the file to Shortcuts, which imports it in the background.
        if on_progress:
            on_progress(f'installing "{name}" (Shortcuts imports asynchronously)')
        subprocess.run(["open", signed], capture_output=True, text=True, timeout=30, check=True)

        deadline = time.time() + _IMPORT_TIMEOUT
        while install_count(name) < 1:
            if time.time() > deadline:
                raise SetupRequiredError(
                    f'Shortcuts did not import "{name}" within {_IMPORT_TIMEOUT:g}s.\n'
                    "If a confirmation panel is open in the Shortcuts app, accept it and "
                    "run setup again.",
                    "setup-cloud",
                    "cloud",
                )
            time.sleep(_IMPORT_POLL)
        if on_progress:
            on_progress(f'"{name}" installed')


def probe_cloud() -> dict[str, Any]:
    """Probe the cloud tier. Never raises."""
    if platform.system() != "Darwin":
        return {
            "available": False,
            "installed": False,
            "reason": f'unsupportedPlatform: this machine reports "{platform.system()}"',
        }
    if not is_apple_silicon_mac():
        return {
            "available": False,
            "installed": False,
            "reason": "deviceNotEligible: Apple Intelligence needs Apple Silicon",
        }
    try:
        subprocess.run(
            ["shortcuts", "list"], capture_output=True, text=True, timeout=20, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "installed": False,
            "reason": "this system does not provide /usr/bin/shortcuts",
        }
    count = install_count()
    if count == 0:
        return {"available": False, "installed": False, "reason": cloud_setup_hint()}
    if count > 1:
        return {
            "available": False,
            "installed": True,
            "reason": _duplicate_message(count, CLOUD_SHORTCUT_NAME),
        }
    return {"available": True, "installed": True, "contextSize": CLOUD_CONTEXT_TOKENS}


def _describe_run_failure(error: Exception, name: str) -> Exception:
    """Turn a ``shortcuts run`` rejection into a typed error."""
    if isinstance(error, subprocess.TimeoutExpired):
        return AppleTimeoutError(
            f"Apple cloud generation got no response within {_CALL_TIMEOUT:g}s. "
            f'Open "{name}" in the Shortcuts app and check that Request is bound to the '
            "Shortcut Input variable -- an unbound Request makes the action wait for a human.",
            "cloud",
        )
    stderr = (getattr(error, "stderr", None) or "").strip()
    if re.search(r"maximum allowed length", stderr, re.IGNORECASE):
        return ContextLengthError(
            f"The prompt exceeded the cloud model's ~{CLOUD_CONTEXT_TOKENS}-token context window.",
            "cloud",
        )
    if re.search(r"QuotaLimitReached|quota", stderr, re.IGNORECASE):
        match = re.search(r"(\d{4}-\d{2}-\d{2}[T ][\d:]+)", stderr)
        reset: Optional[datetime] = None
        if match:
            try:
                reset = datetime.fromisoformat(match.group(1).replace(" ", "T"))
            except ValueError:
                reset = None
        suffix = f" Resets {reset.isoformat()}." if reset else ""
        return QuotaError(
            f"Apple Private Cloud Compute quota reached.{suffix}\n"
            "Fall back to the on-device tier, or try again later.",
            "cloud",
            reset,
        )
    if re.search(r"Couldn.t find shortcut", stderr, re.IGNORECASE):
        return SetupRequiredError(
            f'Shortcuts could not find "{name}".\n{cloud_setup_hint()}', "setup-cloud", "cloud"
        )
    return AppleLLMError(f"Apple cloud generation failed: {stderr or error}", "cloud")


class CloudClient:
    def __init__(self) -> None:
        self._ready = False
        self._quota: Optional[dict[str, Any]] = None

    def set_quota(self, quota: Optional[dict[str, Any]]) -> None:
        """Tell the client what the framework reported about the quota.

        Worth doing because a ``shortcuts run`` against an exhausted quota costs
        a full round trip to find out; this turns that into an immediate typed
        error.
        """
        self._quota = quota

    def _assert_quota(self) -> None:
        if not self._quota or self._quota.get("status") != "limitReached":
            return
        reset = self._quota.get("resetDate")
        parsed: Optional[datetime] = None
        if isinstance(reset, str):
            try:
                parsed = datetime.fromisoformat(reset.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        raise QuotaError(
            "Apple Private Cloud Compute quota is exhausted (reported by the framework "
            "before the call was made).\nFall back to the on-device tier, or try again "
            "later.",
            "cloud",
            parsed,
        )

    @property
    def label(self) -> str:
        return "apple private cloud compute"

    @property
    def context_size(self) -> int:
        return CLOUD_CONTEXT_TOKENS

    def ensure_ready(self, on_progress: Optional[ProgressCallback] = None) -> None:
        if self._ready:
            return
        probe = probe_cloud()
        if not probe.get("available"):
            reason = probe.get("reason") or cloud_setup_hint()
            if not probe.get("installed"):
                raise SetupRequiredError(reason, "setup-cloud", "cloud")
            raise AppleLLMError(reason, "cloud")
        self._ready = True
        if on_progress:
            on_progress(f"apple private cloud compute ready ({CLOUD_CONTEXT_TOKENS} token context)")

    def text(
        self, prompt: str, *, system: Optional[str] = None, web_search: bool = False, **kwargs: Any
    ) -> str:
        if kwargs.get("images"):
            raise AppleLLMError("images needs the on-device tier", "cloud")
        self.ensure_ready()
        self._assert_quota()
        name = CLOUD_SHORTCUT_NAME_WEB if web_search else CLOUD_SHORTCUT_NAME
        if web_search and install_count(name) != 1:
            raise SetupRequiredError(
                f'web_search needs the "{name}" shortcut.\n'
                "Run:  apple-llm setup-cloud --web-search",
                "setup-cloud --web-search",
                "cloud",
            )
        full = f"{system}\n\n{prompt}" if system else prompt

        with tempfile.TemporaryDirectory(prefix="apple-llm-cloud-") as directory:
            in_path = os.path.join(directory, "prompt.txt")
            out_path = os.path.join(directory, "reply.txt")
            with open(in_path, "w", encoding="utf-8") as handle:
                handle.write(full)

            try:
                subprocess.run(
                    ["shortcuts", "run", name, "-i", in_path, "-o", out_path],
                    capture_output=True, text=True, timeout=_CALL_TIMEOUT, check=True,
                )
            except (OSError, subprocess.SubprocessError) as err:
                raise _describe_run_failure(err, name) from err

            try:
                with open(out_path, encoding="utf-8") as handle:
                    reply = handle.read()
            except OSError as err:
                raise AppleLLMError("Apple cloud generation produced no output.", "cloud") from err
            if not reply.strip():
                raise AppleLLMError("Apple cloud generation returned an empty reply.", "cloud")
            return reply

    def json(self, prompt: str, *, schema: JsonSchema, system: Optional[str] = None,
             **kwargs: Any) -> Any:
        """There is no constrained decoding on this tier, so the schema is spelled
        out in the prompt and the reply is mined for JSON. The shape is a request
        here, not a guarantee -- unlike on device."""
        instruction = (
            "Reply with a single JSON value matching this JSON Schema. "
            "Output only the JSON, with no commentary and no code fence.\n\n"
            f"{json.dumps(schema, indent=2)}"
        )
        combined = f"{system}\n\n{instruction}" if system else instruction
        return parse_llm_json(self.text(prompt, system=combined, **kwargs))

    def close(self) -> None:
        """Nothing long-lived: each call is its own ``shortcuts run``."""
