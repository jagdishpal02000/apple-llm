"""A long-lived helper process handling newline-delimited JSON.

This is the single most important piece of the on-device path. Spawning a
helper per request measured ~17s per call against ~1.5s once the model was
resident, with identical prompts and identically short outputs -- so keeping
one process alive is worth roughly an order of magnitude. Both failure modes it
guards against are silent: they produce correct output, just 10-20x slower.

Requests are serialised, and that costs nothing: Apple's framework serialises
inference anyway. Four concurrent requests measured 29.19s against 29.45s
sequentially, so concurrency > 1 buys precisely nothing.
"""

from __future__ import annotations

import atexit
import collections
import json
import queue
import signal
import subprocess
import threading
import weakref
from typing import Callable, Optional

from ._errors import AppleLLMError
from ._errors import TimeoutError as AppleTimeoutError

#: Every live server, so a wedged helper never outlives the interpreter. The
#: Node side does this with `process.once('exit', ...)`.
_live: "weakref.WeakSet[HelperServer]" = weakref.WeakSet()

#: How long a cancelled request may take to say so before the helper is stopped.
_CANCEL_GRACE = 5.0


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
        #: The helper's last stderr lines, quoted when it dies.
        self._stderr: "collections.deque[str]" = collections.deque(maxlen=20)
        self._stderr_reader: Optional[threading.Thread] = None
        #: Serialises callers so one request's reply cannot be handed to another.
        self._lock = threading.Lock()
        #: Control lines arrive from other threads while a request holds _lock.
        self._write_lock = threading.Lock()
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
        self._stderr = collections.deque(maxlen=20)
        process = subprocess.Popen(  # noqa: S603 - path comes from our own cache
            [self._binary, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # The helper speaks UTF-8 whatever the locale says. Left to
            # text=True, a non-UTF-8 locale would misread every non-ASCII reply.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._process = process

        def pump(stream, sink: "queue.Queue[Optional[str]]") -> None:
            try:
                for line in stream:
                    stripped = line.strip()
                    if stripped:
                        sink.put(stripped)
            except (OSError, ValueError):
                pass
            finally:
                # None marks "the child is gone", so a waiting caller is told
                # rather than sitting until its timeout.
                sink.put(None)

        def drain(stream, tail: "collections.deque[str]") -> None:
            # Always read: a pipe nobody reads fills at 64 KB and then blocks
            # the helper mid-write, which looks exactly like a hang.
            try:
                for line in stream:
                    if line.strip():
                        tail.append(line.rstrip("\n"))
            except (OSError, ValueError):
                pass

        self._reader = threading.Thread(target=pump, args=(process.stdout, self._lines), daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=drain, args=(process.stderr, self._stderr), daemon=True)
        self._stderr_reader.start()
        return process

    def _exited(self, process: subprocess.Popen[str]) -> AppleLLMError:
        """The error for a helper that stopped answering, with its last words."""
        try:
            code: Optional[int] = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            code = process.poll()
        reader = self._stderr_reader
        if reader is not None:
            reader.join(timeout=0.5)
        tail = "\n".join(self._stderr).strip()
        self.stop()
        if code is not None and code < 0:
            try:
                how = f"was killed by {signal.Signals(-code).name}"
            except ValueError:
                how = f"was killed by signal {-code}"
        else:
            how = f"exited with code {code}"
        return AppleLLMError(f"Apple helper {how}{': ' + tail[-400:] if tail else ''}", "device")

    def send(self, payload: str, request_id: Optional[str] = None) -> str:
        """One request; returns its reply line.

        Ctrl-C cancels the request in the helper and waits for the helper to say
        so, so its reply can never be handed to the next request instead.
        """
        return self.exchange(payload, request_id=request_id)

    def stream(self, payload: str, on_delta=None, request_id: Optional[str] = None) -> str:
        """Streaming request.

        The helper emits N delta lines plus a final done:true line; each delta
        is forwarded to ``on_delta`` as it arrives and the return value is the
        final line. Holds the server lock for the whole stream, so a stream
        never interleaves with another request -- same guarantee as ``send``.
        """

        def on_event(event: dict) -> None:
            delta = event.get("delta")
            if isinstance(delta, str) and delta and on_delta is not None:
                on_delta(delta)

        return self.exchange(payload, on_event=on_event, request_id=request_id)

    def write(self, control: dict) -> None:
        """Send a control line (endInput, cancel) for the request in flight."""
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            return
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(control) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # The reader reports the real failure.

    def _interrupted(
        self, err: KeyboardInterrupt, already: Optional[KeyboardInterrupt], request_id: Optional[str]
    ) -> KeyboardInterrupt:
        """A Ctrl-C mid-request: ask the helper to stop, and keep reading until it
        says it has. A second Ctrl-C gives up on the helper altogether."""
        if already is not None:
            self.stop()
            raise err
        self.write({"op": "cancel", "id": request_id} if request_id else {"op": "cancel"})
        return err

    def exchange(
        self,
        payload: str,
        on_event: Optional[Callable[[dict], None]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """One request, by the helper's own framing: any number of ``"done": false``
        lines are its events, and the next line is its reply.

        The timeout measures silence rather than total time, so an operation of
        any length is fine while it keeps sending events. Ctrl-C asks the helper
        to stop, because the helper ignores SIGINT, and waits for it to say so:
        otherwise it would go on talking, and the reply would be handed to
        whichever request came next.
        """
        with self._lock:
            process = self._start()
            try:
                assert process.stdin is not None
                with self._write_lock:
                    # json.dumps escapes newlines inside strings, so this only
                    # guards against a caller handing us a literal one.
                    process.stdin.write(payload.replace("\n", " ") + "\n")
                    process.stdin.flush()
            except (BrokenPipeError, ValueError) as err:
                self.stop()
                raise AppleLLMError(f"Apple helper is not accepting input: {err}", "device") from err

            interrupted: Optional[KeyboardInterrupt] = None
            while True:
                try:
                    line = self._lines.get(timeout=self._timeout if interrupted is None else _CANCEL_GRACE)
                except queue.Empty:
                    # A wedged helper cannot be trusted to stay in sync; drop it
                    # so the next request starts from a clean process rather
                    # than reading a reply that belongs to this one.
                    self.stop()
                    if interrupted is not None:
                        raise interrupted from None
                    raise AppleTimeoutError(
                        f"Apple helper timed out after {self._timeout:g}s of silence.", "device"
                    ) from None
                except KeyboardInterrupt as err:
                    interrupted = self._interrupted(err, interrupted, request_id)
                    continue
                if line is None:
                    raise self._exited(process)
                try:
                    parsed = json.loads(line)
                except ValueError:
                    return line  # not JSON: let the caller classify it
                if isinstance(parsed, dict) and parsed.get("done") is False:
                    if on_event is not None and interrupted is None:
                        try:
                            on_event(parsed)
                        except KeyboardInterrupt as err:
                            # Most likely in a callback that prints each delta.
                            interrupted = self._interrupted(err, None, request_id)
                        except Exception:  # noqa: BLE001 - a bad callback must not break the framing
                            pass
                    continue
                if interrupted is not None:
                    raise interrupted
                return line

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
