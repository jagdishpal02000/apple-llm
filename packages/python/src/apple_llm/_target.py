"""Platform detection and Swift toolchain discovery."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional


def is_apple_silicon_mac() -> bool:
    """True when this machine could plausibly run Apple's on-device model.
    Availability itself is confirmed by probing the helper."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def target_triple_from(version_output: str, arch: Optional[str] = None) -> Optional[str]:
    """Build the compile target from an SDK version string, e.g. ``27.0`` ->
    ``arm64-apple-macos27.0``.

    swiftc's own default triple carries a patch component
    (``arm64-apple-macosx27.0.0``) that no shipped stdlib matches, so the target
    has to be spelled out. Pure, so it can be tested against captured ``xcrun``
    output on any machine.
    """
    parts = version_output.strip().split(".")
    if not parts or not parts[0] or not parts[0].isdigit():
        return None
    major = parts[0]
    if len(parts) > 1:
        if not parts[1] or not parts[1].isdigit():
            return None
        minor = parts[1]
    else:
        minor = "0"
    # Normalize so an npm and a pip user on one machine derive the same triple:
    # Node says arm64/x64, Python says arm64/x86_64.
    machine = arch if arch is not None else platform.machine()
    cpu = "x86_64" if machine in ("x64", "x86_64") else "arm64"
    return f"{cpu}-apple-macos{major}.{minor}"


def _run(cmd: list[str], timeout: float = 20.0) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def macos_major() -> Optional[int]:
    """The major macOS version, or None if it cannot be read."""
    out = _run(["sw_vers", "-productVersion"])
    if out is None:
        return None
    head = out.strip().split(".")[0]
    return int(head) if head.isdigit() else None


def host_target() -> Optional[str]:
    """Ask the SDK, then the OS, for a version to build the triple from."""
    for cmd in (["xcrun", "--show-sdk-version"], ["sw_vers", "-productVersion"]):
        out = _run(cmd)
        if out is None:
            continue
        triple = target_triple_from(out)
        if triple is not None:
            return triple
    return None


def find_swiftc() -> Optional[tuple[str, list[str]]]:
    """Locate a Swift compiler as ``(command, prefix_args)``.

    It must be invoked through ``xcrun`` or the ``/usr/bin/swiftc`` shim: those
    set up SDKROOT, whereas the raw binary that ``xcrun -f swiftc`` prints cannot
    find the standard library on its own.
    """
    if _run(["xcrun", "-f", "swiftc"]) is not None:
        return ("xcrun", ["swiftc"])
    if os.path.exists("/usr/bin/swiftc"):
        return ("/usr/bin/swiftc", [])
    return None
