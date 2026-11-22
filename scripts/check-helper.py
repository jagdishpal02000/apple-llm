#!/usr/bin/env python3
"""Build the helper the way a user's first call does, then run it.

    python3 -I scripts/check-helper.py packages/python/src

CI runs this once per installed Xcode, so the helper is proven to compile with
every SDK a user's Command Line Tools might ship, and to load on the runner's
macOS. It goes through the package's own code -- the deployment target, the
compile and the cache -- rather than calling swiftc directly, because that code
is what decides whether a user's first call works.

Exits non-zero if the helper does not build, or does not answer --probe and a
speech request. The on-device model itself needs Apple Intelligence, which CI
runners lack, so it is reported but not required.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "packages/python/src")

from apple_llm._compile import ensure_binary  # noqa: E402
from apple_llm._target import host_target, sdk_version  # noqa: E402


def main() -> int:
    print(f"SDK {sdk_version()}, building for {host_target()}")
    started = time.monotonic()
    binary = ensure_binary(lambda status: print(f"  {status}"))
    print(f"built in {time.monotonic() - started:.0f}s: {binary}")

    probe = json.loads(
        subprocess.run([binary, "--probe"], capture_output=True, text=True, check=True, timeout=120).stdout
    )
    print(json.dumps(probe, indent=2, sort_keys=True))
    assert probe["features"]["protocol"] == 2, "the helper did not describe itself"

    # A request through --serve, which a probe does not exercise. Speech needs
    # no Apple Intelligence, so this works on a runner.
    reply = subprocess.run(
        [binary, "--serve"], input=json.dumps({"op": "voices", "id": "ci"}) + "\n",
        capture_output=True, text=True, timeout=120,
    ).stdout.strip().splitlines()
    assert reply, "the helper did not answer a request on --serve"
    voices = json.loads(reply[-1])
    if voices.get("ok") is True:
        system = (voices.get("systemVoice") or {}).get("name")
        print(f"voices: {len(voices.get('voices', []))}, system voice {system}")
    else:
        # The framing works; the runner's speech setup is not ours to fix.
        print(f"::warning::the voices request failed on this runner: {voices.get('error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
