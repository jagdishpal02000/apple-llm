"""Platform detection and Swift toolchain discovery."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Optional


def is_apple_silicon_mac() -> bool:
    """True when this machine could plausibly run Apple's on-device model.
    Availability itself is confirmed by probing the helper."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def target_triple_from(version_output: str, arch: Optional[str] = None) -> Optional[str]:
    """Build the compile target from a version string, e.g. ``27.0`` ->
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


def _version_parts(text: Optional[str]) -> Optional[tuple[int, int]]:
    """``major.minor`` as two ints, or None when the text is not a version."""
    if text is None:
        return None
    match = re.match(r"(\d+)(?:\.(\d+))?", text.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def deployment_version(sdk: Optional[str], os_version: Optional[str]) -> Optional[str]:
    """The version to build for: the older of the SDK and the running OS.

    The SDK alone is not enough. A Mac on macOS 26 can have the macOS 27 SDK, and
    a binary built for 27 does not run on 26; a Mac on macOS 27 can still have
    the macOS 26 SDK, which cannot target an OS it predates. The helper checks
    every newer API at run time, so building for the older of the two is always
    safe. Pure, so it can be tested on any machine; the Node package has the same
    rule, and the two must agree to share one compiled helper.
    """
    known = [v for v in (_version_parts(sdk), _version_parts(os_version)) if v is not None]
    if not known:
        return None
    major, minor = min(known)
    return f"{major}.{minor}"


def sdk_version() -> Optional[str]:
    """The SDK version the Swift compiler would build against, or None."""
    out = _run(["xcrun", "--show-sdk-version"])
    if out is None or not out.strip():
        return None
    return out.strip()


def host_target() -> Optional[str]:
    """The compile target: the older of the SDK and the running OS (see
    :func:`deployment_version`)."""
    os_out = _run(["sw_vers", "-productVersion"])
    version = deployment_version(sdk_version(), os_out.strip() if os_out else None)
    return None if version is None else target_triple_from(version)


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
