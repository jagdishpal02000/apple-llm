"""A long-lived helper process handling newline-delimited JSON.

This is the single most important piece of the on-device path. api-scribe
measured ~17s per call when it spawned a helper per request against ~1.5s once
the model was resident, with identical prompts and identically short outputs --
so keeping one process alive is worth roughly an order of magnitude. Both failure
modes it guards against are silent: they produce correct output, just 10-20x
slower.

Requests are serialised, and that costs nothing: Apple's framework serialises
inference anyway. Four concurrent requests measured 29.19s against 29.45s
sequentially, so concurrency > 1 buys precisely nothing.
"""

from __future__ import annotations

import atexit
import queue
import subprocess
import threading
import weakref
from typing import Optional

from ._errors import AppleLLMError
from ._errors import TimeoutError as AppleTimeoutError

#: Every live server, so a wedged helper never outlives the interpreter. The
#: Node side does this with `process.once('exit', ...)`.
_live: "weakref.WeakSet[HelperServer]" = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:
    for server in list(_live):
        try:
            server.stop()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


class HelperServer:
    def __init__(self, binary: str, timeout: float = 120.0) -> None:
        self._binary = binary
        self._timeout = timeout
        self._process: Optional[subprocess.Popen[str]] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        #: Serialises callers so one request's reply cannot be handed to another.
        self._lock = threading.Lock()
        _live.add(self)
        # Belt and braces with the atexit hook: a dropped client is cleaned up
        # at collection, without waiting for interpreter shutdown.
        self._finalizer = weakref.finalize(self, _terminate, lambda r=weakref.ref(self): r()._process if r() else None)

    def _start(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        # Drain any stale replies before the new child writes its first.
        # Swap in a fresh queue so an old reader thread's trailing None goes
        # to the old queue instead of poisoning the new child's replies.
        self._lines = queue.Queue()
        process = subprocess.Popen(  # noqa: S603 - path comes from our own cache
            [self._binary, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process

        def pump(stream, sink: "queue.Queue[Optional[str]]") -> None:
            try:
                for line in stream:
                    stripped = line.strip()
                    if stripped:
                        sink.put(stripped)
            finally:
                # None marks "the child is gone", so a waiting caller is told
                # rather than sitting until its timeout.
                sink.put(None)

        self._reader = threading.Thread(
            target=pump, args=(process.stdout, self._lines), daemon=True
        )
        self._reader.start()
        return process

    def send(self, payload: str) -> str:
        with self._lock:
            process = self._start()
            try:
                assert process.stdin is not None
                # json.dumps escapes newlines inside strings, so this only guards
                # against a caller handing us a payload with a literal one.
                process.stdin.write(payload.replace("\n", " ") + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError) as err:
                self.stop()
                raise AppleLLMError(f"Apple helper is not accepting input: {err}", "device") from err

            try:
                line = self._lines.get(timeout=self._timeout)
            except queue.Empty:
                # A wedged helper cannot be trusted to stay in sync; drop it so
                # the next request starts from a clean process rather than
                # reading a reply that belongs to the request that timed out.
                self.stop()
                raise AppleTimeoutError(
                    f"Apple on-device generation timed out after {self._timeout:g}s.", "device"
                ) from None
            if line is None:
                code = process.poll()
                self.stop()
                raise AppleLLMError(f"Apple helper exited with code {code}", "device")
            return line

    def stream(self, payload: str, on_delta=None) -> str:
        """Streaming request.

        The helper emits N delta lines plus a final done:true line; each delta
        is forwarded to ``on_delta`` as it arrives and the return value is the
        final line. Holds the server lock for the whole stream, so a stream
        never interleaves with another request -- same guarantee as ``send``.
        """
        import json as _json

        with self._lock:
            process = self._start()
            try:
                assert process.stdin is not None
                process.stdin.write(payload.replace("\n", " ") + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError) as err:
                self.stop()
                raise AppleLLMError(f"Apple helper is not accepting input: {err}", "device") from err

            for _ in range(10_000):  # bound: a stream is many lines, not infinite
                try:
                    line = self._lines.get(timeout=self._timeout)
                except queue.Empty:
                    self.stop()
                    raise AppleTimeoutError(
                        f"Apple on-device generation timed out after {self._timeout:g}s.",
                        "device",
                    ) from None
                if line is None:
                    code = process.poll()
                    self.stop()
                    raise AppleLLMError(f"Apple helper exited with code {code}", "device")
                try:
                    parsed = _json.loads(line)
                except ValueError:
                    return line  # not JSON: let the caller classify it
                if isinstance(parsed, dict) and parsed.get("ok") is True and not parsed.get("done"):
                    delta = parsed.get("delta")
                    if isinstance(delta, str) and delta and on_delta is not None:
                        try:
                            on_delta(delta)
                        except Exception:  # noqa: BLE001 - a bad callback must not break the stream
                            pass
                    continue
                return line
            self.stop()
            raise AppleLLMError("Apple helper stream did not terminate.", "device")

    def stop(self) -> None:
        process, self._process = self._process, None
        _terminate(lambda: process)


def _terminate(get_process) -> None:
    process = get_process()
    if process is None or process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            process.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass
