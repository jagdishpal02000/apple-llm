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
import time
from pathlib import Path
from typing import Callable, Optional

from ._errors import ModelUnavailableError
from ._target import find_swiftc, host_target, sdk_version

#: Called with a short status string for the one-time compile and the shortcut
#: install. Both take seconds and must not look like a hang.
ProgressCallback = Callable[[str], None]

#: Where to report a helper that will not build.
_ISSUES_URL = "https://github.com/jagdishpal02000/apple-llm/issues"

#: How often a build that is still running says so. A cold first build takes
#: about a minute.
_BUILD_HEARTBEAT = 15.0

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
    A helper edit, an OS upgrade or a new SDK changes it, so both rebuild
    automatically.
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
        # Empty when neither the SDK nor the OS version can be read, exactly as
        # the Node package does, so the two still agree on the key.
        triple = host_target() or ""
        fp = fingerprint(source, triple)
        if (
            _binary_path is not None
            and _binary_fingerprint == fp
            and not force
            and os.path.exists(_binary_path)
        ):
            return _binary_path
        _binary_path = _build(source, triple, on_progress)
        _binary_fingerprint = fp
        return _binary_path


def compiler_errors(output: str) -> str:
    """The compiler's ``error:`` lines, which say what went wrong.

    swiftc's full output leads with dozens of deprecation warnings, so a
    fixed-length prefix of it can cut off before the first error.
    """
    seen: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if (line.startswith("error: ") or " error: " in line) and line not in seen:
            seen.append(line)
    if seen:
        return "\n".join(seen[:6])
    return output.strip()[-600:]


def _build(source: str, triple: str, on_progress: Optional[ProgressCallback]) -> str:
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

    directory.mkdir(parents=True, exist_ok=True)
    unique = f"{os.getpid()}.{threading.get_ident()}.{random.randrange(1 << 30)}"
    source_path = directory / f"{target.name}.{unique}.swift"
    staging = directory / f"{target.name}.{unique}.tmp"
    source_path.write_text(source, encoding="utf-8")

    # A first build takes about a minute on a cold compiler cache, and a silent
    # minute reads as a hang. Say so up front, then keep saying it is working.
    started = time.monotonic()
    done = threading.Event()
    if on_progress:
        on_progress("building the Apple on-device helper (first run only; up to a minute)")

        def heartbeat() -> None:
            while not done.wait(_BUILD_HEARTBEAT):
                try:
                    on_progress(f"still building the helper ({round(time.monotonic() - started)}s)")
                except Exception:  # noqa: BLE001 - a progress callback must not break a build
                    pass

        threading.Thread(target=heartbeat, daemon=True).start()

    command, prefix = swiftc
    target_args = ["-target", triple] if "-apple-" in triple else []
    try:
        subprocess.run(
            [command, *prefix, *target_args, "-parse-as-library", "-O", str(source_path),
             "-o", str(staging)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
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
        output = getattr(err, "stderr", None)
        detail = compiler_errors(output if isinstance(output, str) and output else str(err))
        raise ModelUnavailableError(
            f"Failed to build the Apple on-device helper with {command} "
            f"(SDK {sdk_version() or 'unknown'}, target {triple or 'default'}).\n{detail}\n"
            "It needs macOS 26 or later and the Xcode Command Line Tools to match. Updating "
            "them usually fixes this: System Settings › General › Software Update, or "
            "`xcode-select --install`.\n"
            f"If it still fails, please report it, with this message, at {_ISSUES_URL}",
            "unsupportedOSVersion",
            "device",
        ) from err
    finally:
        done.set()
        source_path.unlink(missing_ok=True)

    if on_progress:
        on_progress(f"helper built in {round(time.monotonic() - started)}s")
    return str(target)
