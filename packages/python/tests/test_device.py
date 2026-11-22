"""The on-device client against a scripted fake helper, so request pairing,
retries and error mapping are tested on any machine."""

import json
import os
import signal
import stat
import sys
import textwrap
import threading

import pytest

from apple_llm import (
    AppleLLMError,
    CancelledError,
    DeviceClient,
    ModelBusyError,
    TimeoutError,
    UnsupportedError,
    restore_nulls,
)


@pytest.fixture
def helper(tmp_path):
    """A fake helper that answers the Nth request with the Nth script entry.

    Each entry is a list of steps: a reply dict, ``{"sleep": s}``,
    ``{"wait": op}`` (block until a control line with that op arrives),
    ``{"stderr": n}`` (write n bytes to stderr) or ``{"exit": code}``. A reply
    with ``"echo": true`` answers with the request's prompt.
    """
    log = tmp_path / "log.jsonl"
    script = tmp_path / "script.json"
    binary = tmp_path / "helper"
    binary.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, signal, sys, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # as the real helper does in --serve
        steps = json.load(open({str(script)!r}))
        log = open({str(log)!r}, "a", buffering=1)
        n = 0
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            message = json.loads(line)
            log.write(line)
            if message.get("op") in ("cancel", "toolResult", "endInput"):
                continue
            for step in steps[min(n, len(steps) - 1)]:
                if "sleep" in step:
                    time.sleep(step["sleep"])
                elif "wait" in step:
                    while True:
                        control = sys.stdin.readline()
                        if not control:
                            sys.exit(0)
                        log.write(control)
                        if json.loads(control).get("op") == step["wait"]:
                            break
                elif "stderr" in step:
                    sys.stderr.write("x" * (step["stderr"] - 1) + "\\n")
                    sys.stderr.flush()
                elif "exit" in step:
                    sys.stderr.write("Fatal error: the helper fell over\\n")
                    sys.stderr.flush()
                    sys.exit(step["exit"])
                else:
                    reply = dict(step)
                    if reply.pop("echo", False):
                        reply["content"] = "answer to " + message.get("prompt", "")
                    print(json.dumps(reply), flush=True)
            n += 1
        """))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    class Helper:
        def client(self, steps) -> DeviceClient:
            script.write_text(json.dumps(steps))
            client = DeviceClient()
            # Skip the probe and the compile: the fake is the helper.
            client._probe = {"available": True}
            client._binary = str(binary)
            return client

        @property
        def requests(self):
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

    return Helper()


ECHO = {"ok": True, "echo": True}


class TestCtrlC:
    def test_the_next_request_gets_its_own_answer(self, helper):
        # Before 0.3.1 an interrupted text() left its reply in the queue, and
        # every later call returned the answer to the call before it.
        client = helper.client([[{"sleep": 1.0}, ECHO], [ECHO]])
        try:
            threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
            with pytest.raises(KeyboardInterrupt):
                client.text("Q1")
            assert client.text("Q2") == "answer to Q2"
            assert client.text("Q3") == "answer to Q3"
        finally:
            client.close()

    def test_it_cancels_the_request_it_interrupted(self, helper):
        client = helper.client([[{"wait": "cancel"}, {"ok": False, "kind": "cancelled", "error": "cancelled"}], [ECHO]])
        try:
            threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
            with pytest.raises(KeyboardInterrupt):
                client.text("a long answer")
            cancels = [r for r in helper.requests if r.get("op") == "cancel"]
            assert cancels and cancels[0]["id"] == helper.requests[0]["id"]
            assert client.text("next") == "answer to next"
        finally:
            client.close()

    def test_a_stream_interrupted_in_its_callback_stays_paired(self, helper):
        delta = {"ok": True, "delta": "word ", "done": False}
        client = helper.client([[delta, {"sleep": 0.2}, delta, {"ok": True, "content": "word word", "done": True}], [ECHO]])

        def on_delta(_text):
            raise KeyboardInterrupt  # what a Ctrl-C inside print() looks like

        try:
            with pytest.raises(KeyboardInterrupt):
                client.stream("story", on_delta=on_delta)
            assert client.text("after") == "answer to after"
        finally:
            client.close()


class TestBusyModel:
    def test_retries_a_request_the_model_manager_turned_away(self, helper):
        busy = {"ok": False, "kind": "busy", "error": "ModelManagerError 1042"}
        client = helper.client([[busy], [busy], [ECHO]])
        try:
            assert client.text("hello") == "answer to hello"
            prompts = [r for r in helper.requests if r.get("prompt") == "hello"]
            assert len(prompts) == 3
            # Each attempt is its own request, with its own id to cancel.
            assert len({r["id"] for r in prompts}) == 3
        finally:
            client.close()

    def test_gives_up_with_model_busy_error(self, helper, monkeypatch):
        import apple_llm._device as device

        monkeypatch.setattr(device, "_BUSY_RETRY_DELAYS", (0.01, 0.01))
        client = helper.client([[{"ok": False, "kind": "busy", "error": "ModelManagerError 1042"}]])
        try:
            with pytest.raises(ModelBusyError):
                client.text("hello")
            assert len([r for r in helper.requests if r.get("prompt") == "hello"]) == 3
        finally:
            client.close()


class TestErrorKinds:
    @pytest.mark.parametrize(
        "kind, error",
        [("unsupported", UnsupportedError), ("timeout", TimeoutError), ("cancelled", CancelledError)],
    )
    def test_maps_each_kind_to_its_class(self, helper, kind, error):
        client = helper.client([[{"ok": False, "kind": kind, "error": "images need macOS 27 or later"}]])
        try:
            with pytest.raises(error):
                client.text("x")
        finally:
            client.close()

    def test_a_crash_names_the_exit_code_and_quotes_stderr(self, helper):
        client = helper.client([[{"exit": 6}]])
        try:
            with pytest.raises(AppleLLMError, match=r"exited with code 6: Fatal error: the helper fell over"):
                client.text("x")
        finally:
            client.close()


class TestStderr:
    def test_a_chatty_helper_does_not_hang(self, helper):
        # 200 KB is three times what an unread pipe holds: without a reader the
        # helper blocks mid-write and the call times out.
        client = helper.client([[{"stderr": 200_000}, ECHO]])
        client._server = None
        try:
            server = client._helper()
            server._timeout = 10
            assert client.text("still here?") == "answer to still here?"
        finally:
            client.close()


class TestNullsRestored:
    def test_pydantic_optional_without_default_validates(self, helper):
        pydantic = pytest.importorskip("pydantic")
        from typing import Optional as Opt

        class Address(pydantic.BaseModel):
            city: str

        class Order(pydantic.BaseModel):
            item: str
            shipping: Opt[Address]
            coupon: Opt[int]

        # What the live model returned for "ebook: Dune": the nullable keys left out.
        client = helper.client([[{"ok": True, "content": '{"item": "ebook: Dune"}'}]])
        try:
            raw = client.json("Extract the order: 'ebook: Dune'.", schema=Order.model_json_schema())
            assert raw == {"item": "ebook: Dune", "shipping": None, "coupon": None}
            assert Order.model_validate(raw).shipping is None
        finally:
            client.close()

    def test_never_overwrites_what_the_model_said(self):
        schema = {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                  "required": ["a"]}
        assert restore_nulls({"a": "kept"}, schema) == {"a": "kept"}

    def test_only_required_nullable_keys_are_filled(self):
        schema = {
            "type": "object",
            "properties": {
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "required_string": {"type": "string"},
                "nullable": {"type": ["integer", "null"]},
            },
            "required": ["required_string", "nullable"],
        }
        assert restore_nulls({"required_string": "x"}, schema) == {"required_string": "x", "nullable": None}

    def test_follows_refs_arrays_and_nesting(self):
        schema = {
            "$defs": {"Item": {"type": "object", "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                               "required": ["note"]}},
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
            "required": ["items"],
        }
        assert restore_nulls({"items": [{}, {"note": "n"}]}, schema) == {"items": [{"note": None}, {"note": "n"}]}
