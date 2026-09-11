"""Typed errors, so callers branch on a class rather than matching a message.

Apple rewords its diagnostics between OS releases; the helper classifies
failures at the Swift boundary and this module names them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

Tier = Literal["device", "cloud"]

#: Why the on-device model cannot be used. Distinguishable, because the fix differs.
UnavailableReason = Literal[
    "appleIntelligenceNotEnabled",
    "modelNotReady",
    "deviceNotEligible",
    "unsupportedPlatform",
    "unsupportedOSVersion",
    "noSwiftCompiler",
    "unknown",
]


class AppleLLMError(Exception):
    """Base class for everything this package raises."""

    def __init__(self, message: str, tier: Optional[Tier] = None) -> None:
        super().__init__(message)
        self.tier = tier


class ModelUnavailableError(AppleLLMError):
    def __init__(
        self, message: str, reason: UnavailableReason = "unknown", tier: Optional[Tier] = None
    ) -> None:
        super().__init__(message, tier)
        self.reason: UnavailableReason = reason


class SchemaRejectedError(AppleLLMError):
    """Apple's ``GenerationSchema`` decoder refused the schema -- almost always a
    dialect rule. See ``_schema.py``."""


class ContextLengthError(AppleLLMError):
    def __init__(
        self, message: str, tier: Optional[Tier] = None, context_size: Optional[int] = None
    ) -> None:
        super().__init__(message, tier)
        self.context_size = context_size


class QuotaError(AppleLLMError):
    """Private Cloud Compute is rate limited, or the on-device model reported
    ``rateLimited``. ``reset_date`` says when to retry, when Apple provides it."""

    def __init__(
        self, message: str, tier: Optional[Tier] = None, reset_date: Optional[datetime] = None
    ) -> None:
        super().__init__(message, tier)
        self.reset_date = reset_date


class TimeoutError(AppleLLMError):  # noqa: A001 - deliberately shadows the builtin here
    """Named to match the Node package. Use ``apple_llm.TimeoutError`` explicitly."""


class SetupRequiredError(AppleLLMError):
    """A one-time setup step has not been run -- currently only the cloud shortcut."""

    def __init__(self, message: str, step: str, tier: Optional[Tier] = None) -> None:
        super().__init__(message, tier)
        self.step = step


class RefusalError(AppleLLMError):
    """The model declined to answer (guardrail or refusal)."""


def unavailable_message(reason: Optional[str]) -> str:
    """Turn an unavailability reason into something the user can act on."""
    base = "Apple's on-device model is not available"
    if reason and "appleIntelligenceNotEnabled" in reason:
        return (
            f"{base}: Apple Intelligence is turned off.\n"
            "Enable it in System Settings > Apple Intelligence & Siri, then try again."
        )
    if reason and "modelNotReady" in reason:
        return (
            f"{base}: the model is still downloading.\n"
            "Leave the Mac online and plugged in for a few minutes, then try again."
        )
    if reason and "deviceNotEligible" in reason:
        return f"{base}: this Mac is not eligible for Apple Intelligence."
    if reason and "unsupportedPlatform" in reason:
        return f"{base}: this platform is not supported (Apple Silicon Mac required)."
    if reason and "unsupportedOSVersion" in reason:
        return (
            f"{base}: the macOS version is not supported.\n"
            "Upgrade to macOS 26 or later, then try again."
        )
    if reason and "noSwiftCompiler" in reason:
        return (
            f"{base}: no Swift compiler was found.\n"
            "Install the Xcode command line tools with `xcode-select --install`, then try again."
        )
    return f"{base}: {reason}" if reason else f"{base}."


def to_unavailable_reason(reason: Optional[str]) -> UnavailableReason:
    """Narrow a helper ``reason`` string to the typed union."""
    if reason:
        for candidate in ("appleIntelligenceNotEnabled", "modelNotReady", "deviceNotEligible", "unsupportedPlatform", "unsupportedOSVersion", "noSwiftCompiler"):
            if candidate in reason:
                return candidate  # type: ignore[return-value]
    return "unknown"
