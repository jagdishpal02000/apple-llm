"""apple-llm CLI -- how people debug this, and how it is tested by hand.

    apple-llm probe
    apple-llm setup-cloud [--web-search] [--force]
    apple-llm run --tier device --system "..." -      # prompt on stdin
    apple-llm run --tier cloud --schema s.json "Extract the fields"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Optional, Sequence

from . import AppleLLM, capture_screenshot, probe, sampling_greedy, sampling_seeded
from ._device import parse_image_flag
from ._cloud import install_cloud_shortcut
from ._errors import AppleLLMError

KNOWN_TOOLS = ("ocr", "barcode", "spotlight")


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
    run.add_argument("--system", help="system instructions")
    run.add_argument("--schema", help="JSON Schema file; ask for JSON matching it")
    run.add_argument("--temperature", type=float,
                     help="default 0.4 (0 degenerates; see the README)")
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--image", action="append", default=[],
                     help="attach an image as path or path::label (repeatable; macOS 27+, device tier)")
    run.add_argument("--document", action="append", default=[],
                     help="inline a text document into the prompt (repeatable, device tier)")
    run.add_argument("--tool", action="append", default=[],
                     help="built-in on-device tool: ocr, barcode, spotlight (repeatable)")
    run.add_argument("--session", help="continue a named conversation across calls")
    run.add_argument("--stream", action="store_true",
                     help="print partials as they arrive (device text only)")
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
    return parser


def _progress(status: str) -> None:
    print(f"  {status}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "probe":
            result = probe(_progress)
            if args.json:
                print(json.dumps(result, indent=2))
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
            return 0 if result["device"].get("available") or result["cloud"].get("available") else 1

        if args.command == "setup-cloud":
            install_cloud_shortcut(_progress, force=args.force, web_search=args.web_search)
            return 0

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
            prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
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
            if args.stream and args.tier == "cloud":
                print("error: --stream needs the on-device tier.", file=sys.stderr)
                return 2
            llm = AppleLLM(
                args.tier,
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
