"""Speech against a scripted helper, so all of it runs on any machine. The real
frameworks are exercised by test_live_speech.py."""

import json
import os
import signal
import stat
import sys
import textwrap
import threading
import time

import pytest

import apple_llm._speech as speech_module
from apple_llm import (
    AppleLLMError,
    AppleSpeech,
    AsyncAppleSpeech,
    CancelledError,
    PermissionDeniedError,
    SetupRequiredError,
    TimeoutError as AppleTimeoutError,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    UnsupportedError,
    to_srt,
    to_vtt,
)
from apple_llm._protocol import HelperServer
from apple_llm._speech import sniff_audio_extension
from apple_llm.cli import main

VOICE = {"id": "com.apple.voice.compact.en-US.Samantha", "name": "Samantha", "language": "en-US", "engine": "avspeech"}
TRANSCRIPT = {
    "ok": True,
    "text": "Hello there.",
    "segments": [{"text": "Hello there.", "start": 0, "end": 1.2, "confidence": 0.9}],
    "language": "en-US",
    "engine": "transcriber",
    "duration": 1.5,
}
VOICES = [
    {"id": "c", "name": "Thomas", "language": "fr-FR", "quality": "default", "gender": "male", "novelty": False,
     "personal": False, "default": False, "engine": "avspeech"},
    {"id": "a", "name": "Samantha", "language": "en-US", "quality": "enhanced", "gender": "female", "novelty": False,
     "personal": False, "default": True, "engine": "avspeech"},
    {"id": "b", "name": "Bells", "language": "en-US", "quality": "default", "gender": "unspecified", "novelty": True,
     "personal": False, "default": False, "engine": "avspeech"},
    {"id": "d", "name": "Aman", "language": "en-IN", "quality": "default", "gender": "male", "novelty": False,
     "personal": False, "default": False, "engine": "say"},
]
SYSTEM = {"id": "a", "name": "Samantha", "language": "en-US", "siri": False, "listed": True}


@pytest.fixture
def helper(tmp_path):
    """A fake helper: an executable that logs each request and replays a script.

    ``script`` is a list of lines to send per request, as JSON or the sentinels
    ``{"sleep": seconds}`` and ``{"wait": op}``, which pause until a control line
    with that op arrives.
    """
    log = tmp_path / "log.jsonl"
    script = tmp_path / "script.json"
    binary = tmp_path / "helper"
    binary.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys, time
        script = json.load(open({str(script)!r}))
        log = open({str(log)!r}, "a", buffering=1)
        def out(message):
            print(json.dumps(message), flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            message = json.loads(line)
            log.write(line)
            op = message.get("op")
            if op in ("cancel", "endInput"):
                continue
            if op == "synthesize" and script.get("write_out", True):
                open(message["out"], "wb").write(b"RIFFaudio")
            for step in script["replies"].get(op, script["replies"].get("*", [])):
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
                else:
                    reply = dict(step)
                    if reply.pop("echo_path", False):
                        reply["path"] = message.get("out")
                    if reply.pop("echo_error_path", False):
                        reply["error"] = "macOS could not read audio from " + message.get("path", "") + "."
                    out(reply)
        """))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    class Helper:
        def client(self, replies, timeout=5.0, write_out=True):
            script.write_text(json.dumps({"replies": replies, "write_out": write_out}))
            return AppleSpeech(server=HelperServer(str(binary), timeout=timeout))

        @property
        def requests(self):
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

        def controls(self, op):
            return [r for r in self.requests if r.get("op") == op]

    return Helper()


def saved_reply(format_="wav"):
    return {"ok": True, "echo_path": True, "format": format_, "duration": 2, "sampleRate": 22050, "bytes": 9, "voice": VOICE}


class TestSpeak:
    def test_sends_only_what_was_asked_for(self, helper):
        with helper.client({"speak": [{"ok": True, "spoken": True, "duration": 2.5, "voice": VOICE}]}) as speech:
            result = speech.speak("Hello", voice="Samantha", speed=1.5)
        assert result.duration == 2.5
        assert result.voice.name == "Samantha"
        request = helper.requests[0]
        assert request["text"] == "Hello" and request["voice"] == "Samantha" and request["speed"] == 1.5
        assert "pitch" not in request and "words" not in request and "ssml" not in request
        assert request["id"]

    def test_asks_for_words_only_when_someone_is_listening(self, helper):
        replies = {"speak": [
            {"ok": True, "done": False, "word": {"text": "Hello", "start": 0.4, "offset": 0, "length": 5}},
            {"ok": True, "spoken": True, "duration": 1, "voice": VOICE},
        ]}
        words = []
        with helper.client(replies) as speech:
            speech.speak("Hello", on_word=words.append)
        assert [w.text for w in words] == ["Hello"]
        assert words[0].end is None
        assert helper.requests[0]["words"] is True


class TestLongOperations:
    """"Long audio" is really a claim about this: the timeout measures silence, and
    a 30 minute read-aloud is only alive to it through heartbeats."""

    def test_survives_far_longer_than_the_timeout_while_heartbeats_keep_coming(self, helper):
        # The timeout has room for the fake helper's own Python to start up.
        beat = [{"sleep": 0.4}, {"ok": True, "heartbeat": True, "done": False}]
        replies = {"speak": beat * 5 + [{"ok": True, "spoken": True, "duration": 1800, "voice": VOICE}]}
        with helper.client(replies, timeout=1.0) as speech:
            assert speech.speak("thirty minutes of text").duration == 1800

    def test_more_events_than_the_old_stream_bound_allows(self, helper):
        # A 30 minute read-aloud with word events is thousands of lines each way.
        words = {"ok": True, "done": False, "word": {"text": "a", "start": 0, "offset": 0, "length": 1}}
        replies = {"speak": [words] * 12_000 + [{"ok": True, "spoken": True, "duration": 1, "voice": VOICE}]}
        seen = []
        with helper.client(replies) as speech:
            speech.speak("a", on_word=seen.append)
        assert len(seen) == 12_000

    def test_still_catches_a_helper_that_goes_quiet(self, helper):
        with helper.client({"speak": [{"sleep": 5}]}, timeout=0.1) as speech:
            with pytest.raises(AppleTimeoutError):
                speech.speak("Hello")

    def test_a_bad_callback_does_not_break_the_framing(self, helper):
        replies = {"speak": [
            {"ok": True, "done": False, "word": {"text": "a", "start": 0, "offset": 0, "length": 1}},
            {"ok": True, "spoken": True, "duration": 1, "voice": VOICE},
        ]}

        def boom(_word):
            raise RuntimeError("bad callback")

        with helper.client(replies) as speech:
            assert speech.speak("a", on_word=boom).duration == 1


class TestSynthesize:
    def test_saves_to_the_path_given_with_home_expanded(self, helper):
        with helper.client({"synthesize": [saved_reply("m4a")]}, write_out=False) as speech:
            saved = speech.synthesize("Hello", out="~/speech.m4a")
        assert saved.path == os.path.join(os.path.expanduser("~"), "speech.m4a")
        assert saved.audio is None and saved.format == "m4a"
        assert "format" not in helper.requests[0]

    def test_returns_bytes_without_a_path_and_cleans_up(self, helper):
        with helper.client({"synthesize": [saved_reply()]}) as speech:
            made = speech.synthesize("Hello", sample_rate=16000)
        assert made.audio == b"RIFFaudio"
        assert made.path is None
        request = helper.requests[0]
        assert request["format"] == "wav" and request["sampleRate"] == 16000
        assert not os.path.exists(os.path.dirname(request["out"]))

    def test_cleans_up_when_the_helper_fails(self, helper):
        failing = {"synthesize": [{"ok": False, "kind": "unsupported", "error": "No MP3 encoder."}]}
        with helper.client(failing, write_out=False) as speech:
            with pytest.raises(UnsupportedError):
                speech.synthesize("Hello", format="mp3")
        assert not os.path.exists(os.path.dirname(helper.requests[0]["out"]))

    def test_carries_word_timings(self, helper):
        reply = {**saved_reply(), "words": [{"text": "Hello", "start": 0, "end": 1, "offset": 0, "length": 5}]}
        with helper.client({"synthesize": [reply]}) as speech:
            made = speech.synthesize("Hello", words=True)
        assert made.words[0].text == "Hello" and made.words[0].end == 1


class TestErrors:
    @pytest.mark.parametrize("kind,expected", [
        ("unsupported", UnsupportedError),
        ("permission", PermissionDeniedError),
        ("cancelled", CancelledError),
        ("speech", AppleLLMError),
    ])
    def test_every_kind_is_a_typed_error(self, helper, kind, expected):
        with helper.client({"speak": [{"ok": False, "kind": kind, "error": "no"}]}) as speech:
            with pytest.raises(expected):
                speech.speak("Hello")

    def test_a_missing_language_names_the_step(self, helper):
        with helper.client({"transcribe": [{"ok": False, "kind": "setup", "language": "de-DE", "error": "Not installed"}]}) as speech:
            with pytest.raises(SetupRequiredError) as caught:
                speech.transcribe(b"RIFF....WAVE")
        assert caught.value.step == "install-speech-language"

    def test_unsupported_is_also_a_value_error(self):
        assert issubclass(UnsupportedError, ValueError)


class TestTranscribe:
    def test_passes_a_path_through(self, helper, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        with helper.client({"transcribe": [TRANSCRIPT]}) as speech:
            result = speech.transcribe(audio, language="en-US", hints=["Kubernetes"], words=True, download=False)
        assert result.text == "Hello there." and result.segments[0].confidence == 0.9
        request = helper.requests[0]
        assert request["path"] == str(audio.resolve())
        assert request["language"] == "en-US" and request["hints"] == ["Kubernetes"]
        assert request["words"] is True and request["download"] is False

    def test_downloading_is_left_to_the_helper_by_default(self, helper):
        with helper.client({"transcribe": [TRANSCRIPT]}) as speech:
            speech.transcribe(b"RIFF....WAVE")
        assert "download" not in helper.requests[0]

    def test_writes_bytes_to_a_temp_file_named_for_what_it_is_then_removes_it(self, helper):
        with helper.client({"transcribe": [TRANSCRIPT]}) as speech:
            speech.transcribe(b"RIFF\0\0\0\0WAVEdata")
            speech.transcribe(bytearray(b"fLaC...."))
            speech.transcribe(memoryview(b"mystery"), filename="call.ogg")
        paths = [r["path"] for r in helper.requests]
        assert [os.path.splitext(p)[1] for p in paths] == [".wav", ".flac", ".ogg"]
        assert not any(os.path.exists(os.path.dirname(p)) for p in paths)

    def test_does_not_show_a_caller_who_gave_it_bytes_a_temp_path(self, helper):
        reply = {"transcribe": [{"ok": False, "kind": "unsupported", "echo_error_path": True}]}
        with helper.client(reply) as speech:
            with pytest.raises(UnsupportedError) as caught:
                speech.transcribe(b"mystery")
        assert str(caught.value) == "macOS could not read audio from these bytes."

    def test_reports_segments_progress_and_a_download(self, helper):
        replies = {"transcribe": [
            {"ok": True, "done": False, "download": {"language": "hi-IN", "fraction": 0.5}},
            {"ok": True, "done": False, "started": {"language": "hi-IN", "engine": "transcriber", "duration": 10}},
            {"ok": True, "done": False, "segment": {"text": "One.", "start": 0, "end": 5}},
            {"ok": True, "done": False, "heartbeat": True},
            {"ok": True, "done": False, "segment": {"text": "Two.", "start": 5, "end": 10}},
            TRANSCRIPT,
        ]}
        segments, progress = [], []
        with helper.client(replies) as speech:
            speech.transcribe(b"RIFF....WAVE", on_segment=segments.append, on_progress=progress.append)
        assert [s.text for s in segments] == ["One.", "Two."]
        assert progress == ["downloading the hi-IN speech model (once) 50%", "transcribing 50%", "transcribing 100%"]


class TestListen:
    OPEN = {"ok": True, "done": False, "listening": True, "language": "en-US"}

    @staticmethod
    def heard(text, final, start=0):
        return {"ok": True, "done": False, "transcript": {"text": text, "final": final, "start": start, "end": start + 1}}

    def test_stops_at_a_pause_by_default(self, helper):
        replies = {"listen": [self.OPEN, self.heard("Hel", False), self.heard("Hello there.", True), TRANSCRIPT]}
        opened = []
        with helper.client(replies) as speech:
            transcript = speech.listen_once(on_listening=lambda: opened.append(True))
        assert transcript.text == "Hello there." and opened == [True]
        assert helper.requests[0]["silence"] == 2.0

    def test_iterates_partial_and_final_events(self, helper):
        replies = {"listen": [self.OPEN, self.heard("Hel", False), self.heard("Hello there.", True), TRANSCRIPT]}
        with helper.client(replies) as speech:
            session = speech.listen()
            events = list(session)
            assert [(e.text, e.final) for e in events] == [("Hel", False), ("Hello there.", True)]
            assert session.wait().language == "en-US"
            assert list(session) == []  # Over is over: a second loop ends at once.

    def test_runs_until_stopped_when_continuous_and_keeps_what_it_heard(self, helper):
        replies = {"listen": [self.OPEN, self.heard("Hello.", True), {"wait": "endInput"}, TRANSCRIPT]}
        with helper.client(replies) as speech:
            session = speech.listen(continuous=True)
            first = next(iter(session))
            assert first.text == "Hello."
            session.stop()
            assert session.wait().text == "Hello there."
        assert "silence" not in helper.requests[0]
        assert helper.controls("endInput")
        assert not helper.controls("cancel")

    def test_honours_a_stop_before_the_microphone_opens(self, helper):
        replies = {"listen": [{"sleep": 0.2}, self.OPEN, {"wait": "endInput"}, TRANSCRIPT]}
        with helper.client(replies) as speech:
            session = speech.listen(continuous=True)
            session.stop()
            assert session.wait(5).text == "Hello there."

    def test_cancel_discards_it(self, helper):
        replies = {"listen": [self.OPEN, {"wait": "cancel"}, {"ok": False, "kind": "cancelled", "error": "cancelled"}]}
        with helper.client(replies) as speech:
            session = speech.listen(continuous=True)
            time.sleep(0.3)
            session.cancel()
            with pytest.raises(CancelledError):
                session.wait(5)

    def test_cancel_before_the_microphone_opens_still_discards_it(self, helper):
        replies = {"listen": [{"sleep": 0.2}, self.OPEN, {"wait": "cancel"}, {"ok": False, "kind": "cancelled", "error": "cancelled"}]}
        with helper.client(replies) as speech:
            session = speech.listen(continuous=True)
            session.cancel()
            with pytest.raises(CancelledError):
                session.wait(5)

    def test_a_context_manager_stops_on_the_way_out(self, helper):
        replies = {"listen": [self.OPEN, {"wait": "endInput"}, TRANSCRIPT]}
        with helper.client(replies) as speech:
            with speech.listen(continuous=True) as session:
                time.sleep(0.2)
            assert session.wait(1).text == "Hello there."

    def test_errors_surface_from_the_loop_and_from_wait(self, helper):
        replies = {"listen": [{"ok": False, "kind": "permission", "error": "Allow your terminal app."}]}
        with helper.client(replies) as speech:
            session = speech.listen()
            with pytest.raises(PermissionDeniedError):
                list(session)
            with pytest.raises(PermissionDeniedError):
                session.wait(1)


class TestVoicesAndLanguages:
    def listing(self, helper):
        return helper.client({"voices": [{"ok": True, "voices": VOICES, "systemVoice": SYSTEM}]})

    def test_sorts_them_and_says_which_is_the_default(self, helper):
        with self.listing(helper) as speech:
            voices = speech.voices()
        assert [v.name for v in voices] == ["Aman", "Bells", "Samantha", "Thomas"]
        assert [v.name for v in voices if v.is_default] == ["Samantha"]

    def test_filters_by_language_family_or_exact_tag_and_hides_novelties(self, helper):
        with self.listing(helper) as speech:
            assert [v.name for v in speech.voices(language="en")] == ["Aman", "Bells", "Samantha"]
            assert [v.name for v in speech.voices(language="en_IN")] == ["Aman"]
            assert [v.name for v in speech.voices(language="en", novelty=False)] == ["Aman", "Samantha"]
            assert speech.voices(language="de") == []

    def test_reports_the_system_voice(self, helper):
        with self.listing(helper) as speech:
            assert speech.system_voice().name == "Samantha"

    def test_lists_installs_and_removes_languages(self, helper):
        replies = {
            "speechLanguages": [{"ok": True, "default": "en-US", "languages": [{"id": "hi-IN", "name": "Hindi", "installed": False, "engine": "transcriber"}]}],
            "installSpeechLanguage": [{"ok": True, "done": False, "download": {"language": "hi-IN", "fraction": 0.25}}, {"ok": True, "language": "hi-IN"}],
            "removeSpeechLanguage": [{"ok": True, "removed": True}],
        }
        progress = []
        with helper.client(replies) as speech:
            listing = speech.languages()
            assert listing.default == "en-US" and listing.languages[0].installed is False
            assert speech.install_language("hi", on_progress=progress.append) == "hi-IN"
            assert speech.remove_language("hi-IN") is True
        assert progress == ["downloading the hi-IN speech model (once) 25%"]


class TestInterrupt:
    def test_ctrl_c_stops_the_helper_and_keeps_the_pairing_intact(self, helper):
        # The real helper ignores SIGINT so a listen can finish; a Ctrl-C has to
        # reach it as a cancel line, and its reply must not go to the next request.
        replies = {"speak": [{"wait": "cancel"}, {"ok": False, "kind": "cancelled", "error": "cancelled"}],
                   "voices": [{"ok": True, "voices": VOICES, "systemVoice": SYSTEM}]}
        with helper.client(replies) as speech:
            threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
            with pytest.raises(KeyboardInterrupt):
                speech.speak("A long story")
            assert helper.controls("cancel")[0]["id"] == helper.requests[0]["id"]
            assert len(speech.voices()) == 4


class TestAsync:
    def test_awaitable_mirror(self, helper):
        import asyncio

        replies = {"voices": [{"ok": True, "voices": VOICES, "systemVoice": SYSTEM}], "transcribe": [TRANSCRIPT]}
        inner = helper.client(replies)

        async def run():
            speech = AsyncAppleSpeech()
            speech._inner = inner
            async with speech:
                voices, transcript = await asyncio.gather(speech.voices(language="fr"), speech.transcribe(b"RIFF....WAVE"))
                return voices, transcript

        voices, transcript = asyncio.run(run())
        assert [v.name for v in voices] == ["Thomas"]
        assert transcript.text == "Hello there."


class TestSubtitles:
    @staticmethod
    def transcript(segments, language="en-US"):
        return Transcript("", segments, language, "transcriber", 0)

    @staticmethod
    def words(texts, step=0.5):
        return [TranscriptWord(t, i * step, (i + 1) * step) for i, t in enumerate(texts)]

    def test_writes_both_with_their_own_timestamp_styles(self):
        # The same strings the Node suite expects, which is what keeps the two honest.
        t = self.transcript([TranscriptSegment("Hello there.", 1.5, 3.25), TranscriptSegment("Bye.", 3661.001, 3662)])
        assert to_srt(t) == "1\n00:00:01,500 --> 00:00:03,250\nHello there.\n\n2\n01:01:01,001 --> 01:01:02,000\nBye.\n"
        assert to_vtt(t) == "WEBVTT\n\n00:00:01.500 --> 00:00:03.250\nHello there.\n\n01:01:01.001 --> 01:01:02.000\nBye.\n"
        assert t.to_srt() == to_srt(t) and t.to_vtt() == to_vtt(t)

    def test_breaks_a_long_sentence_into_cues_short_enough_to_read(self):
        long = [f"word{i}" for i in range(40)]
        cues = to_srt(self.transcript([TranscriptSegment(" ".join(long), 0, 20, words=self.words(long))])).strip().split("\n\n")
        assert len(cues) > 2
        assert all(len(cue.split("\n")[2]) <= 84 for cue in cues)
        assert " ".join(cue.split("\n")[2] for cue in cues) == " ".join(long)

    def test_breaks_on_time_as_well_as_length(self):
        slow = ["one", "two", "three"]
        cues = to_vtt(self.transcript([TranscriptSegment("one two three", 0, 15, words=self.words(slow, 5))])).strip().split("\n\n")[1:]
        assert len(cues) == 3

    def test_no_spaces_between_words_in_languages_that_use_none(self):
        t = self.transcript([TranscriptSegment("你好世界", 0, 1, words=self.words(["你好", "世界"]))], "zh-CN")
        assert "\n你好世界\n" in to_srt(t)

    def test_skips_empty_segments_and_keeps_an_untimed_one_whole(self):
        t = self.transcript([TranscriptSegment("", 0, 1), TranscriptSegment("Whole sentence.", 1, 2)])
        assert to_srt(t) == "1\n00:00:01,000 --> 00:00:02,000\nWhole sentence.\n"


class TestSniffing:
    @pytest.mark.parametrize("magic,extension", [
        (b"RIFF\0\0\0\0WAVE", ".wav"),
        (b"FORM\0\0\0\0AIFF", ".aiff"),
        (b"FORM\0\0\0\0AIFC", ".aiff"),
        (b"caff", ".caf"),
        (b"fLaC", ".flac"),
        (b"\0\0\0\x20ftypM4A ", ".m4a"),
        (b"ID3\x04", ".mp3"),
        (bytes([0xFF, 0xFB, 0x90]), ".mp3"),
        (bytes([0xFF, 0xF1, 0x50]), ".aac"),
        (b"hello world!", None),
        (b"", None),
    ])
    def test_recognises_audio_by_its_bytes(self, magic, extension):
        assert sniff_audio_extension(magic) == extension


class TestPlatform:
    def test_speech_says_why_it_cannot_run_rather_than_failing_to_compile(self, monkeypatch):
        monkeypatch.setattr(speech_module.platform, "system", lambda: "Linux")
        with pytest.raises(AppleLLMError, match="macOS 26 or later on Apple Silicon"):
            AppleSpeech().voices()


class TestCli:
    """The commands, against the scripted helper."""

    @pytest.fixture
    def cli(self, helper, monkeypatch):
        def run(replies, *argv):
            client = helper.client(replies)
            monkeypatch.setattr(speech_module, "_assert_platform", lambda: None)
            monkeypatch.setattr(speech_module, "ensure_binary", lambda *_: client._server._binary)
            return main(list(argv))
        return run

    def test_speak_joins_bare_words_and_says_it(self, cli, helper):
        assert cli({"speak": [{"ok": True, "spoken": True, "duration": 1, "voice": VOICE}]}, "speak", "hello", "there", "--voice", "Samantha") == 0
        request = helper.requests[0]
        assert request["text"] == "hello there" and request["voice"] == "Samantha"

    def test_speak_saves_with_o(self, cli, helper, tmp_path, capsys):
        out = tmp_path / "a.m4a"
        assert cli({"synthesize": [saved_reply("m4a")]}, "speak", "hello", "-o", str(out)) == 0
        assert "wrote" in capsys.readouterr().err
        assert helper.requests[0]["out"] == str(out.resolve())

    def test_speak_with_nothing_to_say_explains(self, cli, monkeypatch, capsys):
        monkeypatch.setattr("apple_llm.cli._piped_input", lambda: None)
        assert cli({}, "speak") == 2
        assert "needs some text" in capsys.readouterr().err

    def test_voices_shows_your_language_and_where_to_change_it(self, cli, capsys):
        assert cli({"voices": [{"ok": True, "voices": VOICES, "systemVoice": SYSTEM}]}, "voices") == 0
        out = capsys.readouterr().out
        assert "Samantha" in out and "your System voice" in out and "System Settings" in out
        assert "Thomas" not in out and "Bells" not in out
        assert "speed only" in out

    def test_voices_all_shows_everything(self, cli, capsys):
        assert cli({"voices": [{"ok": True, "voices": VOICES, "systemVoice": SYSTEM}]}, "voices", "--all") == 0
        out = capsys.readouterr().out
        assert "Thomas" in out and "Bells" in out and "novelty" in out

    def test_transcribe_prints_text(self, cli, capsys):
        assert cli({"transcribe": [TRANSCRIPT]}, "transcribe", "a.wav") == 0
        assert capsys.readouterr().out == "Hello there.\n"

    def test_transcribe_takes_the_format_from_the_extension(self, cli, helper, tmp_path):
        out = tmp_path / "a.srt"
        assert cli({"transcribe": [TRANSCRIPT]}, "transcribe", "a.wav", "-o", str(out)) == 0
        assert out.read_text().startswith("1\n00:00:00,000 --> 00:00:01,200\n")
        assert helper.requests[0]["words"] is True

    def test_transcribe_json(self, cli, capsys):
        assert cli({"transcribe": [TRANSCRIPT]}, "transcribe", "a.wav", "--format", "json") == 0
        assert json.loads(capsys.readouterr().out)["language"] == "en-US"

    def test_a_failure_is_a_message_and_exit_1(self, cli, capsys):
        assert cli({"transcribe": [{"ok": False, "kind": "unsupported", "error": "Audio file not found: a.wav"}]}, "transcribe", "a.wav") == 1
        assert "Audio file not found" in capsys.readouterr().err

    def test_listen_prints_final_text_only(self, cli, capsys):
        replies = {"listen": [
            TestListen.OPEN, TestListen.heard("Hel", False), TestListen.heard("Hello there.", True), TRANSCRIPT,
        ]}
        assert cli(replies, "listen", "--once") == 0
        captured = capsys.readouterr()
        assert captured.out == "Hello there.\n"
        assert "Listening" in captured.err

    def test_languages_marks_what_is_installed_and_installs(self, cli, capsys):
        replies = {
            "speechLanguages": [{"ok": True, "default": "en-US", "languages": [
                {"id": "en-US", "name": "English (United States)", "installed": True, "engine": "transcriber"},
                {"id": "hi-IN", "name": "Hindi (India)", "installed": False, "engine": "transcriber"},
            ]}],
            "installSpeechLanguage": [{"ok": True, "language": "hi-IN"}],
        }
        assert cli(replies, "languages") == 0
        out = capsys.readouterr().out
        assert "* en-US" in out and "(the default)" in out and "  hi-IN" in out
        assert cli(replies, "languages", "install", "hi") == 0
        assert "hi-IN is ready." in capsys.readouterr().out
