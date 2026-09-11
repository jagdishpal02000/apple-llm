"""Compile the Swift helper on first use and cache it.

Never called at install time: importing this package on Linux, an Intel Mac or
macOS 25 must succeed, and the wheel is pure Python with no build step.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from ._errors import ModelUnavailableError
from ._target import find_swiftc, host_target

#: Called with a short status string for the one-time compile and the shortcut
#: install. Both take seconds and must not look like a hang.
ProgressCallback = Callable[[str], None]

_lock = threading.Lock()
_binary_path: Optional[str] = None
_binary_fingerprint: Optional[str] = None
_source_cache: Optional[str] = None


def cache_dir() -> Path:
    """Where the compiled helper is cached.

    Deliberately not namespaced per language binding: the npm and the pip package
    embed byte-identical Swift, so they derive the same fingerprint and share one
    compiled binary. Installing both costs one compile, not two.
    """
    return Path.home() / "Library" / "Caches" / "apple-llm" / "bin"


def fingerprint(source: str, triple: str) -> str:
    """The cache key: sha256 over the source, a newline, and the target triple,
    truncated to 12 hex characters.

    The exact recipe is load-bearing -- the Node package computes the same string
    and must agree byte for byte, or the two would each compile their own copy.
    A helper edit or an OS upgrade changes it, so both rebuild automatically.
    """
    digest = hashlib.sha256(f"{source}\n{triple}".encode()).hexdigest()
    return digest[:12]


def helper_source() -> str:
    """The embedded Swift source, shipped as package data."""
    global _source_cache
    if _source_cache is None:
        _source_cache = (Path(__file__).parent / "swift" / "helper.swift").read_text(
            encoding="utf-8"
        )
    return _source_cache


def ensure_binary(on_progress: Optional[ProgressCallback] = None, force: bool = False) -> str:
    """Return the path to the compiled helper, building it if needed."""
    global _binary_path, _binary_fingerprint
    with _lock:
        source = helper_source()
        triple = host_target() or ""
        fp = fingerprint(source, triple)
        if (
            _binary_path is not None
            and _binary_fingerprint == fp
            and not force
            and os.path.exists(_binary_path)
        ):
            return _binary_path
        _binary_path = _build(on_progress)
        _binary_fingerprint = fp
        return _binary_path


def _build(on_progress: Optional[ProgressCallback]) -> str:
    source = helper_source()
    triple = host_target() or ""
    directory = cache_dir()
    target = directory / f"fm-helper-{fingerprint(source, triple)}"

    if target.exists():
        return str(target)

    swiftc = find_swiftc()
    if swiftc is None:
        raise ModelUnavailableError(
            "apple-llm needs a Swift compiler to build its on-device helper, but none "
            "was found.\nInstall the Xcode command line tools with `xcode-select "
            "--install`, then try again.",
            "noSwiftCompiler",
            "device",
        )

    if on_progress:
        on_progress("building the Apple on-device helper (one time, a few seconds)")
    directory.mkdir(parents=True, exist_ok=True)
    unique = f"{os.getpid()}.{threading.get_ident()}.{random.randrange(1 << 30)}"
    source_path = directory / f"{target.name}.{unique}.swift"
    staging = directory / f"{target.name}.{unique}.tmp"
    source_path.write_text(source, encoding="utf-8")

    command, prefix = swiftc
    target_args = ["-target", triple] if "-apple-" in triple else []
    try:
        subprocess.run(
            [command, *prefix, *target_args, "-parse-as-library", "-O", str(source_path),
             "-o", str(staging)],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        staging.chmod(0o755)
        # Atomic publish, so concurrent first-runs cannot observe a half-written
        # file. os.replace is atomic within a filesystem.
        os.replace(staging, target)
    except (OSError, subprocess.SubprocessError) as err:
        staging.unlink(missing_ok=True)
        # A concurrent run may have finished first, which is a success for us.
        if target.exists():
            return str(target)
        detail = getattr(err, "stderr", None) or str(err)
        raise ModelUnavailableError(
            f"Failed to build the Apple on-device helper with {command}.\n{str(detail)[:400]}\n"
            "This usually means the macOS SDK predates the Foundation Models framework "
            "(macOS 26+).",
            "unsupportedOSVersion",
            "device",
        ) from err
    finally:
        source_path.unlink(missing_ok=True)

    if on_progress:
        on_progress("helper built")
    return str(target)
