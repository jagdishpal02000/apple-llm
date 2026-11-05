"""Speech against the real frameworks: ``APPLE_LLM_LIVE=1 pytest``.

Silent: nothing plays out loud (read-alouds run at volume 0), and nothing here
opens the microphone, so it can run anywhere. Everything is on device.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from apple_llm import (
    AppleLLMError,
    AppleSpeech,
    SetupRequiredError,
    UnsupportedError,
    ensure_binary,
    to_srt,
)
from apple_llm._protocol import HelperServer

pytestmark = pytest.mark.skipif(os.environ.get("APPLE_LLM_LIVE") != "1", reason="set APPLE_LLM_LIVE=1")

SENTENCE = "The quick brown fox jumps over the lazy dog."


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def error_rate(said, heard):
    a, b = words(said), words(heard)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / len(a)


@pytest.fixture(scope="module")
def speech():
    with AppleSpeech() as client:
        yield client


def test_lists_voices_with_one_system_voice(speech):
    voices = speech.voices()
    assert len(voices) > 10
    assert len([v for v in voices if v.is_default]) == 1
    assert speech.system_voice().listed is True


@pytest.mark.parametrize("fmt", ["wav", "aiff", "caf", "m4a", "aac", "flac"])
def test_every_format_reads_back_as_the_same_speech(speech, tmp_path, fmt):
    out = tmp_path / f"round.{fmt}"
    saved = speech.synthesize(SENTENCE, voice="Samantha", out=out)
    assert saved.format == fmt and saved.bytes == out.stat().st_size
    heard = speech.transcribe(out, language="en-US")
    assert error_rate(SENTENCE, heard.text) < 0.25, heard.text


def test_bytes_round_trip_with_word_timings_and_subtitles(speech):
    made = speech.synthesize(SENTENCE, voice="Samantha", words=True)
    assert made.audio[:4] == b"RIFF" and len(made.words) == 9
    heard = speech.transcribe(made.audio, language="en-US", words=True)
    assert error_rate(SENTENCE, heard.text) < 0.25
    assert " --> " in to_srt(heard)
    assert any(s.words for s in heard.segments)


def test_refuses_what_macos_cannot_encode_and_leaves_nothing(speech, tmp_path):
    with pytest.raises(UnsupportedError, match="MP3"):
        speech.synthesize(SENTENCE, out=tmp_path / "nope.mp3")
    assert not list(tmp_path.iterdir())


def test_long_text_renders_whole_with_timings_correct_to_the_end(speech, tmp_path):
    # A zero-length buffer ends a segment, not the text, and marker offsets
    # restart with each segment: both would show up here and nowhere shorter.
    paragraph = "First a sentence to make the engine work. Then another one, a little longer, so that it has more to do. Third!"
    text = "\n\n".join([paragraph] * 120)
    saved = speech.synthesize(text, voice="Samantha", out=tmp_path / "long.m4a", words=True)
    assert saved.duration > 600
    assert len(saved.words) == len(words(text))
    assert all(a.start <= b.start for a, b in zip(saved.words, saved.words[1:]))
    assert saved.words[-1].start > saved.duration - 5


def test_speaks_at_volume_0_with_word_events(speech):
    seen = []
    result = speech.speak(SENTENCE, voice="Samantha", volume=0, on_word=lambda w: seen.append(w.text))
    assert " ".join(seen) == SENTENCE
    assert result.duration > 1.5


def test_ctrl_c_stops_real_speech_and_the_next_request_is_fine(speech):
    # The helper ignores SIGINT, so this only works if the client passes it on.
    started = time.time()
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    with pytest.raises(KeyboardInterrupt):
        speech.speak(" ".join([SENTENCE] * 20), voice="Samantha", volume=0)
    assert time.time() - started < 5
    assert speech.speak("Done.", voice="Samantha", volume=0).voice.name == "Samantha"


def test_says_so_when_macos_stops_speech_nobody_asked_it_to_stop(tmp_path):
    # Delete a running helper's file, as a disk cleaner might, and AVSpeech cancels
    # everything at once. That used to come back as success, with nothing said.
    doomed = tmp_path / "doomed-helper"
    shutil.copy(ensure_binary(), doomed)
    with AppleSpeech(server=HelperServer(str(doomed))) as orphaned:
        orphaned.voices()  # Starts it, while the file still exists.
        doomed.unlink()
        with pytest.raises(AppleLLMError, match="before it finished"):
            orphaned.speak(SENTENCE, voice="Samantha", volume=0)
        with pytest.raises(AppleLLMError, match="before it finished"):
            orphaned.synthesize(SENTENCE, voice="Samantha")


def test_a_missing_language_asks_for_a_download(speech):
    missing = next((l for l in speech.languages().languages if not l.installed), None)
    if missing is None:
        pytest.skip("every language is installed")
    with pytest.raises(SetupRequiredError, match=missing.id):
        speech.transcribe(speech.synthesize("Hello.", voice="Samantha").audio, language=missing.id, download=False)


def test_the_cli_round_trips(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")}

    def cli(*argv):
        return subprocess.run([sys.executable, "-m", "apple_llm.cli", *argv], capture_output=True, text=True, env=env, timeout=120)

    out = tmp_path / "cli.m4a"
    made = cli("speak", "Hello", "from", "the", "command", "line.", "--voice", "Samantha", "-o", str(out))
    assert made.returncode == 0 and out.exists(), made.stderr
    heard = cli("transcribe", str(out), "--language", "en-US")
    assert error_rate("Hello from the command line.", heard.stdout) < 0.25, heard.stdout
    subtitles = tmp_path / "cli.vtt"
    assert cli("transcribe", str(out), "--language", "en-US", "-o", str(subtitles)).returncode == 0
    assert subtitles.read_text().startswith("WEBVTT")
    bad = cli("speak", "Hi", "--voice", "Nobody")
    assert bad.returncode == 1 and "Unknown voice" in bad.stderr
    listing = cli("voices")
    assert listing.returncode == 0 and "System voice" in listing.stdout
