"""apple-llm CLI -- how people debug this, and how it is tested by hand.

    apple-llm probe
    apple-llm setup-cloud [--web-search] [--force]
    apple-llm "What does this error mean?" < error.log   # short for run
    apple-llm run --tier device --system "..." -      # prompt on stdin
    apple-llm run --tier cloud --schema s.json "Extract the fields"
    apple-llm speak "Hello there"                     # out loud
    apple-llm transcribe meeting.m4a --format srt
    apple-llm listen                                  # from the microphone
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import stat
import subprocess
import sys
from typing import Optional, Sequence

from . import AppleLLM, capture_screenshot, probe, sampling_greedy, sampling_seeded
from ._device import parse_image_flag
from ._cloud import install_cloud_shortcut
from ._errors import AppleLLMError
from ._speech import AppleSpeech, Transcript, probe_speech, to_srt, to_vtt

KNOWN_TOOLS = ("ocr", "barcode", "spotlight")

_COMMANDS = {"probe", "setup-cloud", "run", "ask-screen", "history", "reset", "count",
             "rewrite", "proofread", "summarize", "draft",
             "speak", "voices", "transcribe", "listen", "languages"}

_TRANSCRIPT_FORMATS = ("text", "json", "srt", "vtt")
_SYSTEM_VOICE_HINT = 'search System Settings for "System voice"'


def _render_transcript(transcript: Transcript, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(dataclasses.asdict(transcript), indent=2, ensure_ascii=False) + "\n"
    if fmt == "srt":
        return to_srt(transcript)
    if fmt == "vtt":
        return to_vtt(transcript)
    return transcript.text + "\n" if transcript.text else ""


def _transcript_format_for(out: Optional[str]) -> str:
    """``notes.srt`` means subtitles, ``notes.json`` means JSON; anything else is text."""
    extension = os.path.splitext(out)[1][1:].lower() if out else ""
    return extension if extension in _TRANSCRIPT_FORMATS else "text"


class _StatusLine:
    """One status line on stderr, rewritten in place on a terminal. When piped,
    each status appears once with no percentages, so a log is not a wall of updates."""

    def __init__(self) -> None:
        self._tty = sys.stderr.isatty()
        self._active = False
        self._last = ""

    def __call__(self, status: str) -> None:
        if self._tty:
            sys.stderr.write(f"\r\x1b[K  {status}")
            sys.stderr.flush()
            self._active = True
            return
        base = status.rsplit(" ", 1)[0] if status.endswith("%") else status
        if base != self._last:
            print(f"  {base}", file=sys.stderr)
        self._last = base

    def clear(self) -> None:
        if self._active:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()
        self._active = False


def _piped_input() -> Optional[str]:
    """Text piped in beside a prompt -- ``cat notes.md | apple-llm run
    "Summarize this"`` -- or None. Only a pipe or a redirected file is read: a
    terminal is interactive, and other descriptors may never close."""
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError, AttributeError):
        return None
    if not (stat.S_ISFIFO(mode) or stat.S_ISREG(mode)):
        return None
    text = sys.stdin.read()
    return text if text.strip() else None


def _prompt_from(arg: str) -> str:
    """With text piped in, the prompt says what to do with it -- after it, as
    people read it."""
    if arg == "-":
        return sys.stdin.read()
    piped = _piped_input()
    return arg if piped is None else f"{piped.rstrip()}\n\n{arg}"


def _parse_tools(values: list[str]) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        for part in [p.strip() for p in value.split(",") if p.strip()]:
            if part not in KNOWN_TOOLS:
                raise ValueError(f'--tool must be ocr, barcode, or spotlight (got "{part}").')
            if part not in out:
                out.append(part)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apple-llm",
        description="Apple's on-device and Private Cloud Compute models.",
        epilog='apple-llm "<prompt>" is short for run. Text piped into run goes in before '
               'the prompt: cat notes.md | apple-llm "Summarize this"',
    )
    sub = parser.add_subparsers(dest="command")

    probe_cmd = sub.add_parser("probe", help="what this machine can do")
    probe_cmd.add_argument("--json", action="store_true", help="machine-readable output")

    setup = sub.add_parser("setup-cloud", help="install the Shortcut the cloud tier needs")
    setup.add_argument("--web-search", action="store_true",
                       help='also install the "Use Broad World Knowledge" variant')
    setup.add_argument("--force", action="store_true", help="reinstall even if present")

    run = sub.add_parser("run", help="one completion")
    run.add_argument("prompt", help='the prompt, or "-" to read stdin')
    run.add_argument("--tier", choices=("device", "cloud", "auto"), default="auto")
    run.add_argument("--fallback", choices=("device",),
                     help="cloud tier: answer on device when Apple rate-limits the cloud")
    run.add_argument("--system", help="system instructions")
    run.add_argument("--schema", help="JSON Schema file; ask for JSON matching it")
    run.add_argument("--temperature", type=float,
                     help="default 0.4 (0 degenerates; see the README)")
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--image", action="append", default=[],
                     help="attach an image as path or path::label (repeatable; macOS 27+, device tier)")
    run.add_argument("--document", action="append", default=[],
                     help="inline a text document into the prompt (repeatable)")
    run.add_argument("--tool", action="append", default=[],
                     help="built-in on-device tool: ocr, barcode, spotlight (repeatable)")
    run.add_argument("--session", help="continue a named conversation across calls")
    run.add_argument("--stream", action="store_true",
                     help="print text as it arrives (the cloud tier sends it in one piece)")
    run.add_argument("--use-case", choices=("general", "contentTagging"),
                     help="contentTagging is Apple's tagging-specialised model")
    run.add_argument("--guardrails", choices=("default", "permissive"),
                     help="permissive relaxes guardrails for rewriting tasks")
    run.add_argument("--seed", type=int, help="reproducible sampling (top-k, seeded)")
    run.add_argument("--greedy", action="store_true",
                     help="greedy decoding (deterministic, but degenerates)")
    run.add_argument("--web-search", action="store_true", help="cloud tier only")

    ask_screen = sub.add_parser("ask-screen", help="screenshot, then ask (device tier)")
    ask_screen.add_argument("question", help='the question, or "-" to read stdin')
    ask_screen.add_argument("--system", help="system instructions")
    ask_screen.add_argument("--mode", choices=("interactive", "window", "fullscreen"),
                            default="interactive")
    ask_screen.add_argument("--session", help="continue a named conversation across calls")
    ask_screen.add_argument("--max-tokens", type=int)
    ask_screen.add_argument("--image", action="append", default=[])

    history_cmd = sub.add_parser("history", help="show a conversation's mirrored turns")
    history_cmd.add_argument("--session", required=True)
    history_cmd.add_argument("--json", action="store_true", dest="as_json")

    reset_cmd = sub.add_parser("reset", help="drop one conversation, or all")
    reset_cmd.add_argument("--session", default=None)

    for name in ("rewrite", "proofread", "summarize", "draft"):
        preset = sub.add_parser(name, help=f"Write with Siri preset: {name}")
        preset.add_argument("text", help='the text, or "-" to read stdin')
        preset.add_argument("--system", help="system instructions")

    count = sub.add_parser("count", help="tokens this prompt costs, before sending")
    count.add_argument("prompt", help='the prompt, or "-" to read stdin')
    count.add_argument("--system", help="system instructions")
    count.add_argument("--image", action="append", default=[])
    count.add_argument("--tool", action="append", default=[])

    speak_cmd = sub.add_parser("speak", help="say it out loud, or save it with -o file.m4a")
    speak_cmd.add_argument("text", nargs="*", help='the text (no quotes needed), or "-" to read stdin')
    speak_cmd.add_argument("--voice", help="a name or id, or system; default: your System voice")
    speak_cmd.add_argument("--language", "--lang", "-l", help="the best voice for a language: fr, en-GB, hi-IN")
    speak_cmd.add_argument("--speed", type=float, help="1 is normal, 2 is twice as fast (about 0.56 to 4)")
    speak_cmd.add_argument("--pitch", type=float, help="0.5 to 2")
    speak_cmd.add_argument("--volume", type=float, help="0 to 1")
    speak_cmd.add_argument("--ssml", action="store_true", help="the text is SSML")
    speak_cmd.add_argument("--file", help="read the text from a file")
    speak_cmd.add_argument("-o", "--out", help='save instead of playing: .wav .m4a .aac .flac .aiff .caf .pcm ("-" for stdout)')
    speak_cmd.add_argument("--format", help="when the extension does not say")
    speak_cmd.add_argument("--sample-rate", type=float, help="resample, e.g. 16000")
    speak_cmd.add_argument("--words", action="store_true", help="with --json: every word's timing")
    speak_cmd.add_argument("--json", action="store_true", help="print details of what was made")

    voices_cmd = sub.add_parser("voices", help="the voices on this Mac")
    voices_cmd.add_argument("--language", "--lang", "-l", help="en for any English, en-GB for one")
    voices_cmd.add_argument("--all", action="store_true", help="every language, and novelty voices")
    voices_cmd.add_argument("--json", action="store_true")

    transcribe_cmd = sub.add_parser("transcribe", help="audio or video to text, any length")
    transcribe_cmd.add_argument("file", help='the file, or "-" to read audio from stdin')
    transcribe_cmd.add_argument("--language", "--lang", "-l",
                                help="default: this Mac's language; it is not detected from the audio")
    transcribe_cmd.add_argument("--format", choices=_TRANSCRIPT_FORMATS,
                                help="default: from the -o extension, otherwise text")
    transcribe_cmd.add_argument("-o", "--out", help="write there instead of stdout")
    transcribe_cmd.add_argument("--words", action="store_true", help="with --format json: every word's timing")
    transcribe_cmd.add_argument("--hint", action="append", default=[], help="a name or term to expect (repeatable)")
    transcribe_cmd.add_argument("--alternatives", action="store_true", help="with --format json: other readings")
    transcribe_cmd.add_argument("--no-download", action="store_true", help="fail rather than download a missing language")

    listen_cmd = sub.add_parser("listen", help="the microphone to text, live. Ctrl-C stops and keeps what it heard")
    listen_cmd.add_argument("--once", action="store_true", help="stop when you pause")
    listen_cmd.add_argument("--silence", type=float, help="seconds of quiet that end it (default 2 with --once)")
    listen_cmd.add_argument("--max-seconds", type=float, help="stop after this long, speech or no speech")
    listen_cmd.add_argument("--language", "--lang", "-l")
    listen_cmd.add_argument("--hint", action="append", default=[])
    listen_cmd.add_argument("--no-download", action="store_true")
    listen_cmd.add_argument("--words", action="store_true")
    listen_cmd.add_argument("-o", "--out", help="also write the final text there")
    listen_cmd.add_argument("--json", action="store_true", help="one JSON event per line, partial results included")

    languages_cmd = sub.add_parser("languages", help="languages it can recognise")
    languages_cmd.add_argument("action", nargs="?", choices=("install", "remove"))
    languages_cmd.add_argument("code", nargs="?", help="e.g. hi-IN")
    languages_cmd.add_argument("--json", action="store_true")
    return parser


def _progress(status: str) -> None:
    print(f"  {status}", file=sys.stderr)


def _speak_command(args: argparse.Namespace) -> int:
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            text: Optional[str] = handle.read()
    elif args.text == ["-"]:
        text = sys.stdin.read()
    elif args.text:
        text = " ".join(args.text)
    else:
        text = _piped_input()
    if text is None:
        print('speak needs some text, "-" to read stdin, or --file <path>.', file=sys.stderr)
        return 2
    status = _StatusLine()
    options = {"voice": args.voice, "language": args.language, "speed": args.speed, "pitch": args.pitch,
               "volume": args.volume, "ssml": args.ssml, "on_progress": status}
    with AppleSpeech() as speech:
        try:
            to_stdout = args.out == "-" or (args.out is None and args.format is not None)
            if to_stdout:
                if sys.stdout.isatty():
                    print("That would print audio to the terminal. Add -o <file>, or redirect it.", file=sys.stderr)
                    return 2
                made = speech.synthesize(text, format=args.format, sample_rate=args.sample_rate, **options)
                status.clear()
                sys.stdout.buffer.write(made.audio or b"")
                return 0
            if args.out is not None:
                saved = speech.synthesize(text, out=args.out, format=args.format, sample_rate=args.sample_rate,
                                          words=args.words, **options)
                status.clear()
                if args.json:
                    print(json.dumps(dataclasses.asdict(saved), indent=2, ensure_ascii=False))
                else:
                    print(f"wrote {args.out} ({saved.duration:.1f}s, {saved.voice.name})", file=sys.stderr)
                return 0
            spoken = speech.speak(text, **options)
            status.clear()
            if args.json:
                print(json.dumps(dataclasses.asdict(spoken), indent=2, ensure_ascii=False))
            return 0
        finally:
            status.clear()


def _voices_command(args: argparse.Namespace) -> int:
    with AppleSpeech(on_progress=_progress) as speech:
        everything = speech.voices(language=args.language)
        if args.json:
            print(json.dumps([dataclasses.asdict(v) for v in everything], indent=2, ensure_ascii=False))
            return 0
        system = speech.system_voice()
    print(f"Your System voice is {system.name} ({system.language}). To change it, {_SYSTEM_VOICE_HINT}.\n")
    # A hundred and eighty lines is not an answer to "what can I use?".
    family = system.language.split("-")[0]
    shown = everything if args.all else [
        v for v in everything
        if not v.novelty and (args.language is not None or v.language.split("-")[0] == family)
    ]
    if not shown:
        print("No voices match that.")
    width = max([4, *(len(v.name) for v in shown)])
    for voice in shown:
        notes = [voice.quality if voice.quality != "default" else "", "novelty" if voice.novelty else "",
                 "speed only" if voice.engine == "say" else "", "your System voice" if voice.is_default else ""]
        gender = "" if voice.gender == "unspecified" else voice.gender
        print(f"  {voice.name:<{width}}  {voice.language:<7}  {gender:<6}  {', '.join(n for n in notes if n)}".rstrip())
    if not args.all and len(everything) > len(shown):
        print(f"\n{len(everything) - len(shown)} more: --all lists every language and novelty voice, --language fr filters.")
    other = next((v.name for v in shown if not v.is_default), system.name)
    print(f'\nTry one:  apple-llm speak "Hello there" --voice {other}')
    print(f"Better voices are free downloads in the same place, and so are Siri voices where macOS offers them: {_SYSTEM_VOICE_HINT}.")
    return 0


def _languages_command(args: argparse.Namespace) -> int:
    with AppleSpeech() as speech:
        if args.action == "install":
            if not args.code:
                print("languages install needs a language, e.g. hi-IN.", file=sys.stderr)
                return 2
            status = _StatusLine()
            installed = speech.install_language(args.code, on_progress=status)
            status.clear()
            print(f"{installed} is ready.")
            return 0
        if args.action == "remove":
            if not args.code:
                print("languages remove needs a language, e.g. hi-IN.", file=sys.stderr)
                return 2
            removed = speech.remove_language(args.code)
            print(f"Released {args.code}. macOS removes it when it wants the space." if removed
                  else f"{args.code} was not installed by apple-llm, so nothing changed.")
            return 0
        listing = speech.languages()
    if args.json:
        print(json.dumps(dataclasses.asdict(listing), indent=2, ensure_ascii=False))
        return 0
    print("Languages macOS can recognise. A * is already on this Mac; the rest download on first use, in about 25 seconds.\n")
    width = max(len(language.id) for language in listing.languages)
    for language in listing.languages:
        note = "  (the default)" if language.id == listing.default else ""
        print(f"{'*' if language.installed else ' '} {language.id:<{width}}  {language.name}{note}")
    return 0


def _transcribe_command(args: argparse.Namespace) -> int:
    fmt = args.format or _transcript_format_for(args.out)
    source = sys.stdin.buffer.read() if args.file == "-" else args.file
    status = _StatusLine()
    with AppleSpeech() as speech:
        try:
            transcript = speech.transcribe(
                source, language=args.language,
                # Subtitles use word timings to keep each cue short enough to read.
                words=args.words or fmt in ("srt", "vtt"),
                alternatives=args.alternatives, hints=args.hint or None,
                download=not args.no_download, on_progress=status,
            )
        finally:
            status.clear()
    if not transcript.text:
        print("No speech found.", file=sys.stderr)
    rendered = _render_transcript(transcript, fmt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


def _listen_command(args: argparse.Namespace) -> int:
    once = args.once or args.silence is not None
    status = _StatusLine()
    interactive = sys.stderr.isatty()

    def opened() -> None:
        status.clear()
        print("Listening. It stops when you pause." if once else "Listening. Press Ctrl-C to stop.", file=sys.stderr)

    with AppleSpeech() as speech:
        session = speech.listen(
            language=args.language, silence=args.silence, continuous=not once, max_seconds=args.max_seconds,
            words=args.words, hints=args.hint or None, download=not args.no_download,
            on_listening=opened, on_progress=status,
        )
        events = iter(session)
        stopping = False
        while True:
            try:
                event = next(events)
            except StopIteration:
                break
            except KeyboardInterrupt:
                # The first Ctrl-C finishes and keeps the text; a second is for a real hurry.
                if stopping:
                    return 130
                stopping = True
                session.stop()
                continue
            if args.json:
                print(json.dumps(dataclasses.asdict(event), ensure_ascii=False), flush=True)
                continue
            if interactive:
                room = max(20, shutil.get_terminal_size().columns - 1)
                sys.stderr.write("\r\x1b[K" + ("" if event.final else f"\x1b[2m{event.text[-room:]}\x1b[0m"))
                sys.stderr.flush()
            if event.final:
                print(event.text, flush=True)
        transcript = session.wait()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(_render_transcript(transcript, _transcript_format_for(args.out)))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `apple-llm "What does this error mean?"` is a prompt, not a mistyped
    # command: a command never contains a space, and a typo still errors.
    if argv and argv[0] not in _COMMANDS and any(ch.isspace() for ch in argv[0]):
        argv.insert(0, "run")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "probe":
            result = probe(_progress)
            speech = probe_speech()
            if args.json:
                print(json.dumps({**result, "speech": speech}, indent=2))
            else:
                device = result["device"]
                print("on-device")
                print(f"  available   {device.get('available')}")
                if device.get("variant"):
                    print(f"  variant     {device['variant']}")
                if device.get("contextSize"):
                    print(f"  context     {device['contextSize']} tokens")
                caps = device.get("capabilities")
                if caps:
                    on = ", ".join(k for k, v in caps.items() if v) or "none reported"
                    print(f"  supports    {on}")
                if device.get("useCases"):
                    print(f"  use cases   {', '.join(device['useCases'])}")
                if device.get("reason"):
                    print(f"  reason      {device['reason']}")
                cloud = result["cloud"]
                print("\ncloud (private cloud compute)")
                print(f"  available   {cloud.get('available')}")
                print(f"  installed   {bool(cloud.get('installed'))}")
                if cloud.get("contextSize"):
                    print(f"  context     {cloud['contextSize']} tokens")
                quota = cloud.get("quota")
                if quota:
                    approaching = " (approaching limit)" if quota.get("approachingLimit") else ""
                    print(f"  quota       {quota.get('status')}{approaching}")
                    if quota.get("resetDate"):
                        print(f"  resets      {quota['resetDate']}")
                if cloud.get("reason"):
                    print(f"  reason      {cloud['reason'].splitlines()[0]}")
                print("\nspeech")
                print(f"  available   {speech.get('available')}")
                if speech.get("voices"):
                    print(f"  voices      {speech['voices']}")
                if speech.get("system_voice"):
                    print(f"  speaks as   {speech['system_voice']['name']} ({speech['system_voice']['language']})")
                if speech.get("languages"):
                    print(f"  recognises  {speech['languages']} languages, {speech['default_language']} by default")
                if speech.get("reason"):
                    print(f"  reason      {speech['reason'].splitlines()[0]}")
            return 0 if result["device"].get("available") or result["cloud"].get("available") else 1

        if args.command == "setup-cloud":
            install_cloud_shortcut(_progress, force=args.force, web_search=args.web_search)
            return 0

        if args.command == "speak":
            return _speak_command(args)
        if args.command == "voices":
            return _voices_command(args)
        if args.command == "transcribe":
            return _transcribe_command(args)
        if args.command == "listen":
            return _listen_command(args)
        if args.command == "languages":
            return _languages_command(args)

        if args.command == "count":
            prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
            llm = AppleLLM("device")
            try:
                result = llm.count_tokens(prompt, system=args.system,
                                          images=[parse_image_flag(i) for i in args.image] or None,
                                          tools=_parse_tools(args.tool))
            finally:
                llm.close()
            tokens, size = result["tokens"], result["context_size"]
            pct = round(tokens / size * 100) if size else 0
            print(f"{tokens} tokens of {size} ({pct}%)")
            return 1 if tokens > size else 0

        if args.command == "history":
            llm = AppleLLM("device")
            try:
                result = llm.history(args.session)
            finally:
                llm.close()
            if args.as_json:
                print(json.dumps({"session_id": args.session, **result}, indent=2))
            else:
                for turn in result["history"]:
                    who = "you" if turn.get("role") == "user" else "model"
                    print(f"{who}: {turn.get('content')}")
            return 0

        if args.command == "reset":
            llm = AppleLLM("device")
            try:
                llm.reset_session(args.session)
            finally:
                llm.close()
            print(f"reset {args.session}" if args.session else "reset all sessions")
            return 0

        if args.command in ("rewrite", "proofread", "summarize", "draft"):
            text = sys.stdin.read() if args.text == "-" else args.text
            llm = AppleLLM("device", guardrails="permissive", on_progress=_progress)
            try:
                print(getattr(llm, args.command)(text, system=args.system))
            finally:
                llm.close()
            return 0

        if args.command == "ask-screen":
            question = sys.stdin.read() if args.question == "-" else args.question
            print("  select a screen region (Esc cancels)", file=sys.stderr)
            shot = capture_screenshot(args.mode)
            llm = AppleLLM("device", max_tokens=args.max_tokens, on_progress=_progress)
            try:
                images = [parse_image_flag(i) for i in args.image] + [shot]
                print(llm.text(question, system=args.system, session_id=args.session,
                               images=images))
            finally:
                import os as _os

                try:
                    _os.unlink(shot)
                except OSError:
                    pass
                llm.close()
            return 0

        if args.command == "run":
            prompt = _prompt_from(args.prompt)
            sampling = None
            if args.greedy:
                sampling = sampling_greedy()
            elif args.seed is not None:
                sampling = sampling_seeded(args.seed)
            try:
                tools = _parse_tools(args.tool)
            except ValueError as err:
                print(f"error: {err}", file=sys.stderr)
                return 2
            if args.stream and args.schema:
                print("error: --stream is text only; schemas need a complete response.",
                      file=sys.stderr)
                return 2
            llm = AppleLLM(
                args.tier,
                fallback=args.fallback,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_case=args.use_case,
                guardrails=args.guardrails,
                sampling=sampling,
                on_progress=_progress,
            )
            try:
                images = [parse_image_flag(i) for i in args.image] or None
                extra = {"images": images, "documents": args.document or None,
                         "session_id": args.session, "tools": tools,
                         "web_search": args.web_search}
                if args.schema:
                    with open(args.schema, encoding="utf-8") as handle:
                        schema = json.load(handle)
                    result = llm.json(prompt, schema=schema, system=args.system, **extra)
                    print(json.dumps(result, indent=2))
                elif args.stream:
                    def _on_delta(delta: str) -> None:
                        sys.stdout.write(delta)
                        sys.stdout.flush()

                    llm.stream(prompt, system=args.system, on_delta=_on_delta, **extra)
                    sys.stdout.write("\n")
                else:
                    print(llm.text(prompt, system=args.system, **extra))
            finally:
                llm.close()
            return 0
    except (AppleLLMError, OSError, subprocess.SubprocessError, ValueError) as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
