"""Text to speech and speech to text, on device.

    speak("Hello there")
    audio = synthesize("Hello there", voice="Samantha").audio
    text = transcribe("meeting.m4a").text
    said = listen_once().text

The same self-compiled Swift helper as the language model, but none of this
needs the model: it works with Apple Intelligence turned off, and it costs no
quota. Each :class:`AppleSpeech` runs a helper process of its own, because a
helper handles one request at a time and a long transcription must not hold up
generation.

Length is not a limit worth thinking about. Measured on an M4 Air: 60 minutes of
speech rendered in 40 seconds using 62 MB, and 30 minutes transcribed in 28
seconds. A read-aloud as long as you like works too, because the helper sends a
heartbeat every 20 seconds and the wedge timer measures silence.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import queue
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Union

from ._compile import ProgressCallback, ensure_binary
from ._errors import (
    AppleLLMError,
    CancelledError,
    ModelUnavailableError,
    PermissionDeniedError,
    SetupRequiredError,
    UnsupportedError,
)
from ._protocol import HelperServer
from ._target import is_apple_silicon_mac, macos_major

#: Ceiling on helper *silence*. Twenty-second heartbeats keep a long op well inside it.
_SILENCE_TIMEOUT = 120.0

#: The formats macOS can write. MP3, Ogg and Opus it can only read.
AudioFormat = Literal["wav", "aiff", "caf", "m4a", "aac", "flac", "pcm"]

#: Audio to transcribe: a path, or the bytes of an audio file.
AudioInput = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview]


@dataclass(frozen=True)
class Voice:
    """One voice installed on this Mac."""

    #: Stable identifier, e.g. ``com.apple.voice.compact.en-US.Samantha``.
    id: str
    #: What a person calls it, e.g. ``Samantha``. Accepted anywhere a voice is.
    name: str
    #: BCP 47 language tag, e.g. ``en-US``.
    language: str
    #: ``default``, ``enhanced`` or ``premium``. Better voices are free downloads
    #: in System Settings; search it for "System voice".
    quality: str
    gender: str
    #: Sound effects like Bells and Zarvox rather than voices.
    novelty: bool
    personal: bool
    #: The voice chosen in System Settings.
    is_default: bool
    #: ``avspeech`` has word timings, pitch, volume and SSML. ``say`` has speed only,
    #: and can lose words while the language model is busy, so a file takes two renders.
    engine: str


@dataclass(frozen=True)
class VoiceInfo:
    """Which voice actually spoke."""

    id: str
    name: str
    language: str
    engine: str


@dataclass(frozen=True)
class SystemVoice:
    """The "System voice" setting. It can be a Siri voice, which is otherwise
    off limits to apps."""

    id: str
    name: str
    language: str
    siri: bool
    #: False when it is not in :func:`list_voices`, which is how a Siri voice shows up.
    listed: bool


@dataclass(frozen=True)
class SpokenWord:
    """A word and where it falls, in seconds and in the text."""

    text: str
    start: float
    #: None for live ``on_word`` events, which arrive as each word starts.
    end: Optional[float]
    #: Position in the text, in UTF-16 code units.
    offset: int
    length: int


@dataclass(frozen=True)
class SpeakResult:
    #: Seconds it took to say, in real time.
    duration: float
    voice: VoiceInfo


@dataclass(frozen=True)
class Synthesis:
    """Rendered speech: ``audio`` holds the bytes, unless it was saved to ``path``."""

    format: str
    #: Seconds of audio.
    duration: float
    sample_rate: float
    bytes: int
    voice: VoiceInfo
    #: The encoded audio file. Raw 16-bit little-endian mono samples for ``pcm``.
    audio: Optional[bytes] = None
    path: Optional[str] = None
    words: Optional[list[SpokenWord]] = None


@dataclass(frozen=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    #: 0 to 1.
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    #: 0 to 1.
    confidence: Optional[float] = None
    words: Optional[list[TranscriptWord]] = None
    alternatives: Optional[list[str]] = None


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[TranscriptSegment]
    #: The language it was recognised as, e.g. ``en-IN``.
    language: str
    #: ``transcriber`` is Apple's better engine; ``dictation`` covers the
    #: languages it lacks.
    engine: str
    #: Seconds of audio.
    duration: float

    def to_srt(self) -> str:
        """SubRip subtitles. Ask for ``words=True`` to get cues short enough to read."""
        return to_srt(self)

    def to_vtt(self) -> str:
        """WebVTT subtitles. Ask for ``words=True`` to get cues short enough to read."""
        return to_vtt(self)


@dataclass(frozen=True)
class ListenEvent:
    """What :func:`listen` has heard. A ``final=False`` event is replaced by the next one."""

    text: str
    final: bool
    start: float
    end: float
    confidence: Optional[float] = None
    words: Optional[list[TranscriptWord]] = None


@dataclass(frozen=True)
class SpeechLanguage:
    #: BCP 47 tag, e.g. ``hi-IN``.
    id: str
    #: Its name in this Mac's language.
    name: str
    installed: bool
    engine: str


@dataclass(frozen=True)
class SpeechLanguages:
    languages: list[SpeechLanguage]
    #: What transcribe and listen use when not told.
    default: str


def _assert_platform() -> None:
    """Fail before compiling anything, with a reason that is about speech."""
    needs = "Speech needs macOS 26 or later on Apple Silicon"
    if platform.system() != "Darwin":
        raise ModelUnavailableError(
            f'{needs}; this machine reports "{platform.system()}".', "unsupportedPlatform", "device"
        )
    if not is_apple_silicon_mac():
        raise ModelUnavailableError(f"{needs}; this Mac is not Apple Silicon.", "deviceNotEligible", "device")
    major = macos_major()
    if major is not None and major < 26:
        raise ModelUnavailableError(f"{needs}; this is macOS {major}.", "unsupportedOSVersion", "device")


def _raise_for(response: dict[str, Any], raw: str) -> None:
    detail = response.get("error") if isinstance(response.get("error"), str) else raw[:200]
    kind = response.get("kind")
    if kind == "cancelled":
        raise CancelledError("The request was cancelled.", "device")
    if kind == "unsupported":
        raise UnsupportedError(detail, "device")
    if kind == "permission":
        raise PermissionDeniedError(detail, "device")
    if kind == "setup":
        raise SetupRequiredError(detail, "install-speech-language", "device")
    raise AppleLLMError(detail, "device")


def _words_from(raw: Any) -> Optional[list[TranscriptWord]]:
    if not isinstance(raw, list):
        return None
    return [TranscriptWord(w["text"], w["start"], w["end"], w.get("confidence")) for w in raw]


def _segment_from(raw: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        raw["text"], raw["start"], raw["end"], raw.get("confidence"), _words_from(raw.get("words")),
        raw.get("alternatives"),
    )


def _transcript_from(response: dict[str, Any]) -> Transcript:
    return Transcript(
        response["text"], [_segment_from(s) for s in response.get("segments", [])],
        response["language"], response["engine"], response["duration"],
    )


def _voice_info(raw: dict[str, Any]) -> VoiceInfo:
    return VoiceInfo(raw["id"], raw["name"], raw["language"], raw["engine"])


def _synthesis_from(response: dict[str, Any], audio: Optional[bytes], saved: bool) -> Synthesis:
    words = response.get("words")
    return Synthesis(
        format=response["format"],
        duration=response["duration"],
        sample_rate=response["sampleRate"],
        bytes=response["bytes"],
        voice=_voice_info(response["voice"]),
        audio=audio,
        path=response["path"] if saved else None,
        words=None if not isinstance(words, list) else [
            SpokenWord(w["text"], w["start"], w.get("end"), w["offset"], w["length"]) for w in words
        ],
    )


def _matches_language(tag: str, wanted: str) -> bool:
    have = tag.lower()
    want = wanted.replace("_", "-").lower()
    return have == want or have.startswith(want + "-")


def sniff_audio_extension(data: bytes) -> Optional[str]:
    """A file extension from magic bytes, so macOS opens in-memory audio as what it is."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return ".aiff"
    if data[:4] == b"caff":
        return ".caf"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[4:8] == b"ftyp":
        return ".m4a"
    if data[:3] == b"ID3":
        return ".mp3"
    if len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        # ADTS AAC and MPEG audio share a sync word; the layer bits tell them apart.
        return ".aac" if data[1] & 0x06 == 0 else ".mp3"
    return None


class _OnDisk:
    """A path to the audio, writing in-memory audio to a private temp file for the call."""

    def __init__(self, source: AudioInput, filename: Optional[str]) -> None:
        self._directory: Optional[str] = None
        self.temporary = False
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
            self._directory = tempfile.mkdtemp(prefix="apple-llm-audio-")
            named = os.path.splitext(filename)[1] if filename else ""
            self.path = os.path.join(self._directory, "audio" + (sniff_audio_extension(data) or named or ".audio"))
            with open(self.path, "wb") as handle:
                handle.write(data)
            self.temporary = True
        else:
            self.path = str(Path(os.fspath(source)).expanduser().resolve())

    def __enter__(self) -> "_OnDisk":
        return self

    def __exit__(self, *exc: object) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)


class ListenSession:
    """One :meth:`AppleSpeech.listen`.

    Iterate it for live events; :meth:`wait` returns the transcript.
    :meth:`stop` finishes and keeps what was heard; :meth:`cancel` discards it.
    Used as a context manager, it stops on the way out.
    """

    def __init__(self, start: Callable[["ListenSession"], Transcript]) -> None:
        self._events: "queue.Queue[Optional[ListenEvent]]" = queue.Queue()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._result: Optional[Transcript] = None
        self._error: Optional[BaseException] = None
        self._stop_requested = False
        self._cancel_requested = False
        self._send_stop: Optional[Callable[[], None]] = None
        self._send_cancel: Optional[Callable[[], None]] = None

        def run() -> None:
            try:
                self._result = start(self)
            except BaseException as err:  # noqa: BLE001 - handed to whoever waits
                self._error = err
            finally:
                self._done.set()
                self._events.put(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _opened(self, send_stop: Callable[[], None], send_cancel: Callable[[], None]) -> None:
        """The microphone is open, so there is something to stop."""
        with self._lock:
            self._send_stop = send_stop
            self._send_cancel = send_cancel
            stop, cancel = self._stop_requested, self._cancel_requested
        # Asked to finish before the microphone even opened.
        if cancel:
            send_cancel()
        elif stop:
            send_stop()

    def stop(self) -> None:
        """Stop listening. The session still finishes, with everything heard so far."""
        with self._lock:
            self._stop_requested = True
            send = self._send_stop
        if send is not None:
            send()

    def cancel(self) -> None:
        """Stop listening and discard it: :meth:`wait` raises ``CancelledError``."""
        with self._lock:
            self._cancel_requested = True
            send = self._send_cancel
        if send is not None:
            send()

    def __iter__(self) -> Iterator[ListenEvent]:
        while True:
            event = self._events.get()
            if event is None:
                self._events.put(None)  # A second iterator ends at once, not never.
                if self._error is not None:
                    raise self._error
                return
            yield event

    def wait(self, timeout: Optional[float] = None) -> Transcript:
        """Block until it is over and return everything it heard."""
        if not self._done.wait(timeout):
            raise TimeoutError("listen did not finish in time; call stop() first.")
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def __enter__(self) -> "ListenSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
        self._done.wait(10)


class AppleSpeech:
    """The speech API with a helper process of its own.

    The module-level :func:`speak`, :func:`transcribe` and friends share one of
    these, which is enough for most programs. Create your own when something must
    not wait: a helper handles one request at a time, so a long transcription
    holds up a ``speak`` on the same instance.
    """

    def __init__(
        self, *, on_progress: Optional[ProgressCallback] = None, server: Optional[HelperServer] = None
    ) -> None:
        self._on_progress = on_progress
        self._server = server
        self._lock = threading.Lock()

    def ensure_ready(self, on_progress: Optional[ProgressCallback] = None) -> None:
        """Compile the helper now rather than on the first call."""
        self._helper(on_progress)

    def _helper(self, on_progress: Optional[ProgressCallback] = None) -> HelperServer:
        with self._lock:
            if self._server is None:
                _assert_platform()
                binary = ensure_binary(on_progress or self._on_progress)
                self._server = HelperServer(binary, timeout=_SILENCE_TIMEOUT)
            return self._server

    def _call(
        self,
        envelope: dict[str, Any],
        *,
        on_progress: Optional[ProgressCallback] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        on_request_id: Optional[Callable[[str, HelperServer], None]] = None,
    ) -> dict[str, Any]:
        server = self._helper(on_progress)
        report = on_progress or self._on_progress
        request_id = uuid.uuid4().hex
        if on_request_id is not None:
            on_request_id(request_id, server)

        def handle(event: dict[str, Any]) -> None:
            download = event.get("download")
            if isinstance(download, dict):
                if report is not None:
                    percent = round(float(download.get("fraction", 0)) * 100)
                    report(f"downloading the {download.get('language')} speech model (once) {percent}%")
                return
            if on_event is not None:
                on_event(event)

        payload = {k: v for k, v in envelope.items() if v is not None}
        raw = server.exchange(json.dumps({**payload, "id": request_id}), handle, request_id=request_id)
        try:
            response = json.loads(raw)
        except ValueError as err:
            raise AppleLLMError(f"Apple helper returned an unreadable reply: {raw[:200]}", "device") from err
        if response.get("ok") is not True:
            _raise_for(response, raw)
        return response

    def voices(self, *, language: Optional[str] = None, novelty: bool = True) -> list[Voice]:
        """Every voice on this Mac, sorted by language and then name.

        ``language`` is ``en`` for any English or ``en-GB`` for one. ``novelty=False``
        leaves out sound effects like Bells and Zarvox.
        """
        response = self._call({"op": "voices"})
        out = [
            Voice(v["id"], v["name"], v["language"], v["quality"], v["gender"], v["novelty"], v["personal"],
                  v["default"], v["engine"])
            for v in response["voices"]
        ]
        if language is not None:
            out = [v for v in out if _matches_language(v.language, language)]
        if not novelty:
            out = [v for v in out if not v.novelty]
        return sorted(out, key=lambda v: (v.language, v.name))

    def system_voice(self) -> SystemVoice:
        """The voice chosen in System Settings, which is what speech uses unless told otherwise."""
        raw = self._call({"op": "voices"})["systemVoice"]
        return SystemVoice(raw["id"], raw["name"], raw["language"], raw["siri"], raw["listed"])

    def speak(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        ssml: bool = False,
        on_word: Optional[Callable[[SpokenWord], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> SpeakResult:
        """Say it out loud, returning when it has finished.

        ``voice`` is a name (``Samantha``), an id from :func:`list_voices`, or
        ``system`` to go through macOS ``say`` with whatever the System voice is.
        The default is the System voice. ``language`` picks the best voice for a
        language. ``speed`` is a multiplier, about 0.56 to 4; ``pitch`` is 0.5 to
        2; ``volume`` is 0 to 1.

        Ctrl-C stops it mid-word. ``on_word`` is called as each word starts; most
        ``avspeech`` voices send these, and some send none.
        """

        def handle(event: dict[str, Any]) -> None:
            word = event.get("word")
            if word is not None and on_word is not None:
                on_word(SpokenWord(word["text"], word["start"], word.get("end"), word["offset"], word["length"]))

        response = self._call(
            {"op": "speak", "text": text, "voice": voice, "language": language, "speed": speed, "pitch": pitch,
             "volume": volume, "ssml": ssml or None, "words": True if on_word is not None else None},
            on_progress=on_progress,
            on_event=handle,
        )
        return SpeakResult(response["duration"], _voice_info(response["voice"]))

    def synthesize(
        self,
        text: str,
        *,
        out: Optional[Union[str, "os.PathLike[str]"]] = None,
        format: Optional[str] = None,  # noqa: A002 - the word everyone uses for it
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        ssml: bool = False,
        sample_rate: Optional[float] = None,
        words: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Synthesis:
        """Render speech to audio: a file with ``out``, otherwise ``.audio`` holds the bytes.

        ``format`` is wav, aiff, caf, m4a, aac, flac or pcm; by default it comes
        from the extension of ``out``, otherwise wav. macOS cannot encode MP3.

        For long text prefer ``out``: thirty minutes is 80 MB as WAV and 11 MB as
        M4A, and ``.audio`` holds all of it in memory.
        """
        fields = {
            "op": "synthesize", "text": text, "voice": voice, "language": language, "speed": speed,
            "pitch": pitch, "volume": volume, "ssml": ssml or None, "sampleRate": sample_rate,
            "words": words or None,
        }
        if out is not None:
            path = str(Path(os.fspath(out)).expanduser().resolve())
            return _synthesis_from(self._call({**fields, "out": path, "format": format}, on_progress=on_progress), None, True)
        chosen = format or "wav"
        directory = tempfile.mkdtemp(prefix="apple-llm-speech-")
        try:
            path = os.path.join(directory, f"speech.{chosen}")
            response = self._call({**fields, "out": path, "format": chosen}, on_progress=on_progress)
            with open(path, "rb") as handle:
                return _synthesis_from(response, handle.read(), False)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def transcribe(
        self,
        source: AudioInput,
        *,
        language: Optional[str] = None,
        words: bool = False,
        alternatives: bool = False,
        hints: Optional[list[str]] = None,
        download: bool = True,
        filename: Optional[str] = None,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Transcript:
        """Transcribe a recording: audio or video, any length, nothing leaves the machine.

        ``source`` is a path or the bytes of an audio file. ``language`` is
        ``en-US`` or just ``en``; the default is this Mac's language, because macOS
        needs it named and does not detect one. ``hints`` are words to expect that
        it might otherwise miss. A missing language downloads on first use, in
        about 25 seconds, unless ``download=False``.
        """
        report = on_progress or self._on_progress
        total = {"duration": 0.0}

        def handle(event: dict[str, Any]) -> None:
            started = event.get("started")
            if isinstance(started, dict):
                total["duration"] = float(started.get("duration") or 0)
            raw = event.get("segment")
            if raw is None:
                return
            segment = _segment_from(raw)
            if on_segment is not None:
                on_segment(segment)
            if report is not None and total["duration"] > 0:
                report(f"transcribing {min(100, round(segment.end / total['duration'] * 100))}%")

        with _OnDisk(source, filename) as disk:
            try:
                response = self._call(
                    {"op": "transcribe", "path": disk.path, "language": language, "words": words or None,
                     "alternatives": alternatives or None, "hints": hints or None,
                     "download": None if download else False},
                    on_progress=on_progress,
                    on_event=handle,
                )
            except AppleLLMError as err:
                if disk.temporary and disk.path in str(err):
                    # The caller gave us bytes. A temp path means nothing to them.
                    raise type(err)(str(err).replace(disk.path, "these bytes"), *_error_args(err)) from None
                raise
        return _transcript_from(response)

    def listen(
        self,
        *,
        language: Optional[str] = None,
        silence: Optional[float] = None,
        continuous: bool = False,
        max_seconds: Optional[float] = None,
        words: bool = False,
        hints: Optional[list[str]] = None,
        download: bool = True,
        on_listening: Optional[Callable[[], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ListenSession:
        """Listen to the microphone.

        By default it ends once you stop talking, after ``silence`` seconds of
        quiet (2 unless told). With ``continuous=True`` it goes on until
        :meth:`ListenSession.stop`::

            with listen(continuous=True) as session:
                for event in session:
                    print(event.text, event.final)

        macOS asks once for microphone access, on behalf of the app that launched
        this: Terminal, VS Code, your own.
        """

        def start(session: ListenSession) -> Transcript:
            ids: dict[str, Any] = {}

            def remember(request_id: str, server: HelperServer) -> None:
                ids["id"], ids["server"] = request_id, server

            def handle(event: dict[str, Any]) -> None:
                if event.get("listening") is True:
                    server: HelperServer = ids["server"]
                    session._opened(
                        lambda: server.write({"op": "endInput"}),
                        lambda: server.write({"op": "cancel", "id": ids["id"]}),
                    )
                    if on_listening is not None:
                        on_listening()
                    return
                heard = event.get("transcript")
                if isinstance(heard, dict):
                    session._events.put(ListenEvent(
                        heard["text"], heard["final"], heard["start"], heard["end"], heard.get("confidence"),
                        _words_from(heard.get("words")),
                    ))

            response = self._call(
                {"op": "listen", "language": language, "silence": None if continuous else (2.0 if silence is None else silence),
                 "maxSeconds": max_seconds, "words": words or None, "hints": hints or None,
                 "download": None if download else False},
                on_progress=on_progress,
                on_event=handle,
                on_request_id=remember,
            )
            return _transcript_from(response)

        return ListenSession(start)

    def listen_once(self, **kwargs: Any) -> Transcript:
        """Listen until you stop talking and return what you said."""
        return self.listen(**kwargs).wait()

    def languages(self) -> SpeechLanguages:
        """Languages macOS can recognise, and which are already on this Mac."""
        response = self._call({"op": "speechLanguages"})
        return SpeechLanguages(
            [SpeechLanguage(l["id"], l["name"], l["installed"], l["engine"]) for l in response["languages"]],
            response["default"],
        )

    def install_language(self, language: str, *, on_progress: Optional[ProgressCallback] = None) -> str:
        """Download a language's model now rather than on first use. Returns its tag."""
        return self._call({"op": "installSpeechLanguage", "language": language}, on_progress=on_progress)["language"]

    def remove_language(self, language: str) -> bool:
        """Say this helper no longer needs a language. macOS deletes it when it wants the space."""
        return self._call({"op": "removeSpeechLanguage", "language": language}).get("removed") is True

    def close(self) -> None:
        """Shut the helper process down. Safe to call more than once."""
        with self._lock:
            if self._server is not None:
                self._server.stop()
                self._server = None

    def __enter__(self) -> "AppleSpeech":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _error_args(err: AppleLLMError) -> tuple:
    """The rest of an error's constructor arguments, to rebuild it with a new message."""
    if isinstance(err, SetupRequiredError):
        return (err.step, err.tier)
    return (err.tier,)


class AsyncAppleSpeech:
    """The same API, awaitable.

    Each call runs the blocking client in a worker thread. A helper handles one
    request at a time anyway, so a thread is the honest shape, and it avoids a
    second copy of the protocol that could drift from the sync one.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._inner = AppleSpeech(**kwargs)

    async def voices(self, **kwargs: Any) -> list[Voice]:
        return await asyncio.to_thread(lambda: self._inner.voices(**kwargs))

    async def system_voice(self) -> SystemVoice:
        return await asyncio.to_thread(self._inner.system_voice)

    async def speak(self, text: str, **kwargs: Any) -> SpeakResult:
        return await asyncio.to_thread(lambda: self._inner.speak(text, **kwargs))

    async def synthesize(self, text: str, **kwargs: Any) -> Synthesis:
        return await asyncio.to_thread(lambda: self._inner.synthesize(text, **kwargs))

    async def transcribe(self, source: AudioInput, **kwargs: Any) -> Transcript:
        return await asyncio.to_thread(lambda: self._inner.transcribe(source, **kwargs))

    async def listen_once(self, **kwargs: Any) -> Transcript:
        return await asyncio.to_thread(lambda: self._inner.listen_once(**kwargs))

    async def languages(self) -> SpeechLanguages:
        return await asyncio.to_thread(self._inner.languages)

    async def install_language(self, language: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(lambda: self._inner.install_language(language, **kwargs))

    async def remove_language(self, language: str) -> bool:
        return await asyncio.to_thread(lambda: self._inner.remove_language(language))

    def close(self) -> None:
        self._inner.close()

    async def __aenter__(self) -> "AsyncAppleSpeech":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self._inner.close)


_shared: Optional[AppleSpeech] = None
_shared_lock = threading.Lock()


def _default_speech() -> AppleSpeech:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AppleSpeech()
        return _shared


def speak(text: str, **kwargs: Any) -> SpeakResult:
    """See :meth:`AppleSpeech.speak`."""
    return _default_speech().speak(text, **kwargs)


def synthesize(text: str, **kwargs: Any) -> Synthesis:
    """See :meth:`AppleSpeech.synthesize`."""
    return _default_speech().synthesize(text, **kwargs)


def list_voices(**kwargs: Any) -> list[Voice]:
    """See :meth:`AppleSpeech.voices`."""
    return _default_speech().voices(**kwargs)


def system_voice() -> SystemVoice:
    """See :meth:`AppleSpeech.system_voice`."""
    return _default_speech().system_voice()


def transcribe(source: AudioInput, **kwargs: Any) -> Transcript:
    """See :meth:`AppleSpeech.transcribe`."""
    return _default_speech().transcribe(source, **kwargs)


def listen(**kwargs: Any) -> ListenSession:
    """See :meth:`AppleSpeech.listen`."""
    return _default_speech().listen(**kwargs)


def listen_once(**kwargs: Any) -> Transcript:
    """See :meth:`AppleSpeech.listen_once`."""
    return _default_speech().listen_once(**kwargs)


def list_speech_languages() -> SpeechLanguages:
    """See :meth:`AppleSpeech.languages`."""
    return _default_speech().languages()


def install_speech_language(language: str, **kwargs: Any) -> str:
    """See :meth:`AppleSpeech.install_language`."""
    return _default_speech().install_language(language, **kwargs)


def remove_speech_language(language: str) -> bool:
    """See :meth:`AppleSpeech.remove_language`."""
    return _default_speech().remove_language(language)


def probe_speech(on_progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    """What speech can do on this machine. Never raises, and works with Apple Intelligence off."""
    speech = AppleSpeech(on_progress=on_progress)
    try:
        voices = speech.voices()
        system = speech.system_voice()
        languages = speech.languages()
        return {
            "available": True,
            "voices": len(voices),
            "system_voice": {"id": system.id, "name": system.name, "language": system.language, "siri": system.siri},
            "languages": len(languages.languages),
            "installed_languages": [l.id for l in languages.languages if l.installed],
            "default_language": languages.default,
        }
    except (AppleLLMError, OSError) as err:
        return {"available": False, "reason": str(err)}
    finally:
        speech.close()


#: Scripts with no spaces between words, where joining words with one would be wrong.
_UNSPACED = re.compile(r"^(zh|ja|yue|th|lo|km|my)\b", re.IGNORECASE)


def _caption_cues(transcript: Transcript) -> list[tuple[float, float, str]]:
    """Cues short enough to read: under 84 characters and 7 seconds where word timings allow it."""
    separator = "" if _UNSPACED.match(transcript.language) else " "
    cues: list[tuple[float, float, str]] = []
    for segment in transcript.segments:
        if not segment.text:
            continue
        if not segment.words:
            cues.append((segment.start, segment.end, segment.text))
            continue
        group: list[TranscriptWord] = []

        def flush() -> None:
            if group:
                cues.append((group[0].start, group[-1].end, separator.join(w.text for w in group)))
                group.clear()

        for word in segment.words:
            length = sum(len(w.text) + len(separator) for w in group) + len(word.text)
            if group and (length > 84 or word.end - group[0].start > 7):
                flush()
            group.append(word)
        flush()
    return cues


def _clock(seconds: float, separator: str) -> str:
    total = max(0, round(seconds * 1000))
    return (
        f"{total // 3_600_000:02d}:{total // 60_000 % 60:02d}:{total // 1000 % 60:02d}"
        f"{separator}{total % 1000:03d}"
    )


def to_srt(transcript: Transcript) -> str:
    """SubRip subtitles. Ask ``transcribe`` for ``words=True`` to get cues short enough to read."""
    return "\n".join(
        f"{index}\n{_clock(start, ',')} --> {_clock(end, ',')}\n{text}\n"
        for index, (start, end, text) in enumerate(_caption_cues(transcript), start=1)
    )


def to_vtt(transcript: Transcript) -> str:
    """WebVTT subtitles. Ask ``transcribe`` for ``words=True`` to get cues short enough to read."""
    cues = [
        f"{_clock(start, '.')} --> {_clock(end, '.')}\n{text}\n"
        for start, end, text in _caption_cues(transcript)
    ]
    return "\n".join(["WEBVTT\n", *cues])
