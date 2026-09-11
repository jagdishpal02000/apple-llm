"""Tests that need the hardware.

Gated on an env var *and* a runtime probe, so a machine without Apple
Intelligence skips cleanly rather than failing:

    APPLE_LLM_LIVE=1 pytest
"""

import asyncio
import json
import os
import statistics
import subprocess
import time

import pytest
from conftest import IMAGE_DIR, load_fixtures

from apple_llm import (
    AppleLLM,
    AppleLLMError,
    AsyncAppleLLM,
    DeviceClient,
    SchemaRejectedError,
    cache_dir,
    ensure_binary,
    fingerprint,
    helper_source,
    host_target,
    probe,
    probe_device,
    sampling_seeded,
    to_apple_schema,
)

LIVE = os.environ.get("APPLE_LLM_LIVE") == "1"
#: Cloud calls burn a shared quota, so they need their own opt-in.
LIVE_CLOUD = os.environ.get("APPLE_LLM_LIVE_CLOUD") == "1"

pytestmark = pytest.mark.skipif(not LIVE, reason="set APPLE_LLM_LIVE=1 to run live tests")


@pytest.fixture(scope="module")
def device_available():
    return bool(probe_device().get("available"))


@pytest.fixture(scope="module")
def client(device_available):
    if not device_available:
        pytest.skip("the on-device model is not available on this machine")
    instance = DeviceClient()
    yield instance
    instance.close()


def test_probe_reports_a_context_window_and_variant(device_available):
    if not device_available:
        pytest.skip("no on-device model")
    result = probe()
    assert result["device"]["available"] is True
    assert result["device"]["contextSize"] >= 4096
    assert isinstance(result["device"]["variant"], str)


def test_returns_free_text(client):
    out = client.text(
        "Name exactly one primary colour. Reply with the word only.",
        system="You are terse.",
        max_tokens=30,
    )
    assert out.strip()
    assert len(out) < 200


def test_honours_the_schema_not_merely_the_request(client):
    # Constrained decoding makes this a guarantee. If it ever fails, the schema
    # transform is wrong -- the model cannot disobey a schema it was given.
    out = client.json(
        "Ada Lovelace, born 1815, worked in mathematics.",
        schema={
            "type": "object",
            "required": ["name", "born", "field"],
            "properties": {
                "name": {"type": "string"},
                "born": {"type": "integer"},
                "field": {"enum": ["mathematics", "physics", "poetry"]},
            },
        },
        max_tokens=200,
    )
    assert isinstance(out["name"], str)
    assert isinstance(out["born"], int)
    assert out["field"] in ("mathematics", "physics", "poetry")
    assert sorted(out) == ["born", "field", "name"]


@pytest.mark.parametrize("fixture", load_fixtures(), ids=[f["name"] for f in load_fixtures()])
def test_apple_accepts_every_corpus_schema(client, fixture):
    """The strongest check available: Apple's own GenerationSchema decoder is the
    authority on the dialect, so the corpus is validated against it and not just
    against this implementation."""
    client.complete(
        "Produce a minimal example.",
        schema=to_apple_schema(fixture["input"]),
        max_tokens=120,
    )


def test_median_call_stays_under_4s(client):
    # This is the regression guard for the session-reuse trap. A fresh process
    # per request measured ~17s a call against ~1.5s once resident, and both
    # failure modes produce *correct* output -- only slower -- so nothing else
    # would catch it. Medians, never a single run: wall-clock swings.
    latencies = []
    for i in range(5):
        started = time.time()
        client.text(f"Name one primary colour. Attempt {i}.", max_tokens=20)
        latencies.append(time.time() - started)
    median = statistics.median(latencies)
    print(f"  median {median:.2f}s of {[round(v, 2) for v in latencies]}")
    assert median < 4.0


def test_a_raw_schema_is_rejected_as_schema_rejected_error(client):
    with pytest.raises(SchemaRejectedError):
        # Raw JSON Schema, not run through to_apple_schema: no title, no x-order.
        client.complete("hi", schema={"type": "object", "properties": {"a": {"type": "string"}}})


def test_compiles_from_a_cold_cache(device_available):
    if not device_available:
        pytest.skip("no on-device model")
    target = cache_dir() / f"fm-helper-{fingerprint(helper_source(), host_target() or '')}"
    target.unlink(missing_ok=True)

    progress = []
    rebuilt = ensure_binary(progress.append, force=True)
    assert rebuilt == str(target)
    # The one-time compile takes seconds and must be reported, not look like a hang.
    assert any("building" in line for line in progress)
    result = subprocess.run([rebuilt, "--probe"], capture_output=True, text=True, timeout=30)
    assert json.loads(result.stdout)["available"] is True


def test_pydantic_models_come_back_validated(device_available):
    if not device_available:
        pytest.skip("no on-device model")
    pydantic = pytest.importorskip("pydantic")

    class Person(pydantic.BaseModel):
        name: str
        born: int

    with AppleLLM(tier="device") as llm:
        person = llm.json("Ada Lovelace, born 1815.", schema=Person, max_tokens=120)
    assert isinstance(person, Person)
    assert person.born == 1815
    assert "Lovelace" in person.name


def test_async_client_works(device_available):
    if not device_available:
        pytest.skip("no on-device model")

    async def run():
        async with AsyncAppleLLM(tier="device") as llm:
            first, second = await asyncio.gather(
                llm.text("Name one primary colour. One word.", max_tokens=20),
                llm.text("Name one primary colour. One word.", max_tokens=20),
            )
            return first, second

    first, second = asyncio.run(run())
    assert first.strip() and second.strip()


def _helper_pids():
    result = subprocess.run(
        ["bash", "-c", 'pgrep -f "fm-helper-.* --serve" || true'],
        capture_output=True, text=True,
    )
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


def test_close_leaves_no_helper_running(device_available):
    if not device_available:
        pytest.skip("no on-device model")
    # Other clients in this file legitimately hold a helper open, so compare
    # pids rather than asserting none is running at all.
    before = _helper_pids()
    llm = AppleLLM(tier="device")
    llm.text("hi", max_tokens=10)
    started = _helper_pids() - before
    assert started

    llm.close()
    time.sleep(0.5)
    assert not (started & _helper_pids())


@pytest.mark.skipif(not LIVE_CLOUD, reason="set APPLE_LLM_LIVE_CLOUD=1 (consumes quota)")
def test_cloud_round_trip():
    if not probe()["cloud"].get("available"):
        pytest.skip("the cloud shortcut is not installed")
    with AppleLLM(tier="cloud") as llm:
        assert "pong" in llm.text("Reply with the single word: pong").lower()


class TestMacOS27Capabilities:
    """Capabilities macOS 27 added on top of what api-scribe used. Each was
    confirmed against the framework before being wired up; see the README."""

    def test_reports_what_the_model_can_do(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        device = probe()["device"]
        caps = device.get("capabilities")
        # Absent on macOS 26 -- the framework only reports capabilities from 27.
        if caps is None:
            pytest.skip("capabilities need macOS 27")
        assert caps["guidedGeneration"] is True
        assert isinstance(caps["vision"], bool)
        assert isinstance(caps["reasoning"], bool)
        assert "contentTagging" in device.get("useCases", [])

    def test_reads_pcc_quota_without_calling_it(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        quota = probe()["cloud"].get("quota")
        if quota is None:
            pytest.skip("quota needs macOS 27")
        # The entitlement that blocks PCC inference does not block reading this.
        assert quota["status"] in ("belowLimit", "limitReached", "unknown")
        assert isinstance(quota["isAvailable"], bool)

    def test_counts_tokens_before_sending(self, client):
        short = client.count_tokens("The quick brown fox jumps over the lazy dog.")
        assert 5 < short["tokens"] < 20
        assert short["context_size"] >= 4096

        # Instructions share the window, so they have to be counted too.
        with_system = client.count_tokens(
            "The quick brown fox jumps over the lazy dog.",
            system="You are a helpful assistant that always answers very briefly indeed.",
        )
        assert with_system["tokens"] > short["tokens"]

        assert client.count_tokens("word " * 3000)["tokens"] > 2000

    def test_prewarm_leaves_the_model_usable(self, client):
        client.prewarm("You are terse.")
        assert client.text(
            "Name one primary colour. One word.", system="You are terse.", max_tokens=20
        ).strip()

    def test_seeded_sampling_is_reproducible(self, client):
        # This is the answer to the temperature-0 trap: determinism without
        # greedy decoding's degeneration. It works only because sessions are
        # fresh per request -- reusing one changes the transcript, and the output.
        runs = {
            client.text(
                "Invent a name for a coffee shop.",
                system="You write one short sentence.",
                temperature=0.9,
                max_tokens=40,
                sampling=sampling_seeded(42),
            )
            for _ in range(3)
        }
        assert len(runs) == 1

    def test_seeded_sampling_survives_a_prewarm(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        # Regression: prewarming used to leave a warmed session that the next
        # call reused, so the first call after a prewarm ran against different
        # session state than the ones after it -- identical seeds, different
        # text, no error.
        fresh = DeviceClient()
        try:
            fresh.prewarm()
            runs = {
                fresh.text(
                    "Invent a name for a coffee shop.",
                    temperature=0.9,
                    max_tokens=30,
                    sampling=sampling_seeded(7),
                )
                for _ in range(3)
            }
        finally:
            fresh.close()
        assert len(runs) == 1

    def test_describes_an_image(self, client):
        if not probe()["device"].get("capabilities", {}).get("vision"):
            pytest.skip("this model has no vision capability")
        image = str(IMAGE_DIR / "red-square-blue-circle.png")
        out = client.text(
            "What shape and colours are in this image? One sentence.",
            system="You describe images literally and briefly.",
            images=[image],
            max_tokens=80,
        ).lower()
        assert "blue" in out
        assert "circle" in out or "round" in out

    def test_missing_image_is_rejected_not_ignored(self, client):
        with pytest.raises(AppleLLMError, match="image not found"):
            client.text("describe", images=["/definitely/not/here.png"])

    def test_content_tagging_use_case(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        tagger = DeviceClient(use_case="contentTagging")
        try:
            out = tagger.json(
                "A recipe for sourdough bread using a rye starter and a cast iron dutch oven.",
                schema={
                    "type": "object",
                    "required": ["tags"],
                    "properties": {
                        "tags": {"type": "array", "maxItems": 5, "items": {"type": "string"}}
                    },
                },
                max_tokens=80,
            )
        finally:
            tagger.close()
        assert out["tags"]
        assert any(w in " ".join(out["tags"]).lower() for w in ("sourdough", "bread", "rye"))

    def test_permissive_guardrails_for_rewriting(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        permissive = DeviceClient(guardrails="permissive")
        try:
            assert permissive.text(
                "Rewrite more formally: gonna grab a bite",
                system="You rewrite text.",
                max_tokens=50,
            ).strip()
        finally:
            permissive.close()

    def test_streams_text_with_deltas_that_concatenate(self, client):
        deltas: list[str] = []
        out = client.stream(
            "Count to three, one per line.",
            system="You are terse.",
            max_tokens=40,
            on_delta=deltas.append,
        )
        assert out.strip()
        assert "".join(deltas) == out

    def test_named_conversation_remembers_then_resets(self, device_available):
        import time as _time

        if not device_available:
            pytest.skip("no on-device model")
        session_id = f"live-py-{int(_time.time() * 1000)}"
        owned = DeviceClient()
        try:
            owned.text(
                "My cat is called Biscuit. Reply with one word.",
                system="You are terse and you remember names.",
                session_id=session_id,
                max_tokens=30,
            )
            recall = owned.text(
                "What is my cat called? One word.",
                system="You are terse and you remember names.",
                session_id=session_id,
                max_tokens=30,
            )
            assert "biscuit" in recall.lower()

            history = owned.history(session_id)
            assert len(history["history"]) >= 4
            assert history["history"][0]["role"] == "user"
        finally:
            owned.reset_session(session_id)
            history = owned.history(session_id)
            owned.close()
        assert history["history"] == []

    def test_labelled_image_attachment(self, client):
        if not probe()["device"].get("capabilities", {}).get("vision"):
            pytest.skip("this model has no vision capability")
        image = str(IMAGE_DIR / "red-square-blue-circle.png")
        out = client.text(
            "What is in the image labelled chart? One sentence.",
            system="You describe images literally and briefly.",
            images=[{"path": image, "label": "chart"}],
            max_tokens=80,
        ).lower()
        assert "blue" in out

    def test_builtin_ocr_tool(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        if not probe()["device"].get("capabilities", {}).get("toolCalling"):
            pytest.skip("this model has no tool calling capability")
        image = str(IMAGE_DIR / "red-square-blue-circle.png")
        owned = DeviceClient()
        try:
            out = owned.text(
                'Read any text in this image. If there is none, say "none".',
                system="You read text in images.",
                images=[image],
                tools=["ocr"],
                max_tokens=80,
            )
        finally:
            owned.close()
        assert out.strip()

    def test_write_with_siri_rewrite_preset(self, device_available):
        if not device_available:
            pytest.skip("no on-device model")
        with AppleLLM(tier="device") as llm:
            assert llm.rewrite("gonna grab a bite", instruction="Make it formal.").strip()
