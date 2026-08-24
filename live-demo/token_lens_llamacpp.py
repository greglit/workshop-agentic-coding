#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Explore and edit a Gemma 4 token path through llama.cpp."""

from __future__ import annotations

import argparse
import http.server
import json
import math
import os
import queue
import re
import select
import sys
import termios
import threading
import time
import tty
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests

RESET = "\033[0m"
BOLD = "\033[1m"
GREY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BLACK = "\033[30m"
BG_HIGHLIGHT = "\033[48;5;250m"
INVERSE = "\033[7m"
ALT_SCREEN_ON = "\033[?1049h"
ALT_SCREEN_OFF = "\033[?1049l"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"
CLEAR_SCREEN = "\033[2J\033[H"

SPECIAL_RE = re.compile(r"(<\|[^>]+>|<[^>]+\|>|<bos>|<eos>)")
TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\{(?P<args>.*?)\}<tool_call\|>",
    re.DOTALL,
)
QUOTED_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> str:
    weather = {
        "Berlin": "18 °C, bewölkt",
        "Hamburg": "16 °C, Regen",
        "München": "21 °C, sonnig",
        "Munich": "21 °C, sunny",
    }
    return weather.get(city, f"No weather data for {city}")


TOOL_IMPL = {"get_weather": get_weather}


class ServerError(RuntimeError):
    pass


class LlamaCpp:
    def __init__(self, host: str, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.host}{path}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, requests.JSONDecodeError) as exc:
            raise ServerError(f"{path}: {exc}") from exc

    def health(self) -> None:
        try:
            response = self.session.get(f"{self.host}/health", timeout=3)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ServerError(str(exc)) from exc

    def apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        return str(
            self.post(
                "/apply-template",
                {
                    "messages": messages,
                    "tools": TOOLS if tools is None else tools,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )["prompt"]
        )

    def tokenize(self, text: str) -> list[int]:
        tokens = self.post(
            "/tokenize",
            {"content": text, "add_special": False, "with_pieces": False},
        )["tokens"]
        return [int(token["id"] if isinstance(token, dict) else token) for token in tokens]

    def detokenize(self, token_ids: list[int]) -> str:
        return str(self.post("/detokenize", {"tokens": token_ids})["content"])

    def candidates(
        self, token_ids: list[int], args: argparse.Namespace, seed: int
    ) -> tuple[int, list["Candidate"]]:
        data = self.post(
            "/completion",
            {
                "prompt": token_ids,
                "n_predict": 1,
                "n_probs": args.top,
                "return_tokens": True,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": seed,
            },
        )
        rows = data.get("completion_probabilities") or []
        if not rows:
            raise ServerError("/completion returned no token probabilities")
        chosen = rows[0]
        sampled_id = int(chosen.get("id", data.get("tokens", [None])[0]))
        candidates = [Candidate.from_api(item) for item in chosen.get("top_logprobs") or []]
        if not any(candidate.token_id == sampled_id for candidate in candidates):
            candidates.insert(0, Candidate.from_api(chosen))
        return sampled_id, candidates


@dataclass(frozen=True)
class Candidate:
    token_id: int
    piece: str
    probability: float

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Candidate":
        probability = (
            float(data["prob"])
            if "prob" in data
            else math.exp(float(data.get("logprob", -100.0)))
        )
        return cls(int(data["id"]), str(data.get("token", "")), probability)


@dataclass
class Snapshot:
    kind: str
    token_ids: list[int]
    generated_ids: list[int]
    label: str
    highlight_piece: str = ""
    highlight_context: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    selected: int = 0
    sampled_id: int | None = None
    detail: str = ""


class TerminalScreen:
    def __init__(self, *, alternate: bool = True) -> None:
        self.interactive = sys.stdout.isatty() and sys.stdin.isatty()
        self.alternate = alternate
        self.settings: list[Any] | None = None

    def __enter__(self) -> "TerminalScreen":
        if self.interactive:
            self.settings = termios.tcgetattr(sys.stdin.fileno())
            if self.alternate:
                print(ALT_SCREEN_ON + CLEAR_SCREEN, end="", flush=True)
            print(CURSOR_HIDE, end="", flush=True)
            self.key_mode()
        return self

    def key_mode(self) -> None:
        if self.interactive:
            tty.setcbreak(sys.stdin.fileno())
            print(CURSOR_HIDE, end="", flush=True)

    def line_mode(self) -> None:
        if self.interactive and self.settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.settings)
            print(CURSOR_SHOW, end="", flush=True)

    def clear(self) -> None:
        if self.interactive and self.alternate:
            print(CLEAR_SCREEN, end="", flush=True)

    def __exit__(self, *_: object) -> None:
        if self.interactive:
            if self.settings is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.settings)
            print(CURSOR_SHOW, end="", flush=True)
            if self.alternate:
                print(ALT_SCREEN_OFF, end="", flush=True)


def read_key(timeout: float | None = None) -> str | None:
    if timeout is not None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
    first = os.read(sys.stdin.fileno(), 1)
    if first != b"\x1b":
        return first.decode("utf-8", "ignore").lower()
    ready, _, _ = select.select([sys.stdin], [], [], 0.03)
    if not ready:
        return "escape"
    second = os.read(sys.stdin.fileno(), 1)
    if second != b"[":
        return "escape"
    third = os.read(sys.stdin.fileno(), 1)
    return {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(
        third, "escape"
    )


def heading(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}\n")


def render_context_text(
    text: str, *, thinking: bool = False, highlighted: bool = False
) -> tuple[str, bool]:
    rendered = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if SPECIAL_RE.fullmatch(part):
            style = f"{BG_HIGHLIGHT}{MAGENTA}" if highlighted else MAGENTA
            rendered.append(f"{style}{part}{RESET}")
            if part == "<|channel>":
                thinking = True
            elif part == "<channel|>":
                thinking = False
            continue
        if highlighted:
            style = f"{BG_HIGHLIGHT}{BLACK}"
        elif thinking:
            style = GREY
        else:
            style = ""
        rendered.append(f"{style}{part}{RESET}")
    return "".join(rendered), thinking


def split_tail(text: str, tail: str) -> tuple[str, str]:
    if tail and text.endswith(tail):
        return text[: -len(tail)], tail
    return text, ""


def render_snapshot(client: LlamaCpp, snapshot: Snapshot, auto: bool) -> None:
    heading("CURRENT MODEL CONTEXT")
    context = client.detokenize(snapshot.token_ids)
    highlighted = snapshot.highlight_context or snapshot.highlight_piece
    start = context.rfind(highlighted) if highlighted else -1
    if start >= 0:
        before = context[:start]
        newest = highlighted
        after = context[start + len(highlighted) :]
    else:
        before, newest, after = context, "", ""
    rendered, thinking = render_context_text(before)
    print(rendered, end="")
    if newest:
        rendered, thinking = render_context_text(
            newest, thinking=thinking, highlighted=True
        )
        print(rendered, end="")
    if after:
        rendered, _ = render_context_text(after, thinking=thinking)
        print(rendered, end="")
    print()

    if snapshot.label:
        heading(snapshot.label)
    if snapshot.detail:
        print(f"{YELLOW}{snapshot.detail}{RESET}")
    if snapshot.candidates:
        for index, candidate in enumerate(snapshot.candidates):
            cursor = ">" if index == snapshot.selected else " "
            sampled = " sampled" if candidate.token_id == snapshot.sampled_id else ""
            colour = YELLOW if index == snapshot.selected else GREY
            label = json.dumps(candidate.piece, ensure_ascii=False)
            bar = "█" * max(0, round(candidate.probability * 24))
            print(
                f"{colour}{cursor} {label:<22} {candidate.probability:9.4%} "
                f"{bar}{sampled}{RESET}"
            )
    if snapshot.kind != "done":
        auto_action = "pause" if auto else "play"
        print(
            f"\n{GREY}[←] undo  [→] commit/next  [↑/↓] select  "
            f"[a] {auto_action}  [q] quit{RESET}",
            flush=True,
        )


def render_persistent_snapshot(
    client: LlamaCpp,
    snapshot: Snapshot,
    auto: bool,
    *,
    show_context: bool = False,
) -> None:
    if show_context:
        heading("CURRENT MODEL CONTEXT")
        rendered, _ = render_context_text(client.detokenize(snapshot.token_ids))
        print(rendered)

    token_number = len(snapshot.generated_ids) + 1
    if snapshot.kind == "done":
        heading("ASSISTANT TURN COMPLETE")
        rendered, _ = render_context_text(client.detokenize(snapshot.generated_ids))
        print(rendered)
        return

    heading(f"NEXT TOKEN {token_number}")
    if snapshot.highlight_piece:
        print(f"{GREY}previous:{RESET} {YELLOW}{snapshot.highlight_piece!r}{RESET}\n")
    for index, candidate in enumerate(snapshot.candidates):
        cursor = ">" if index == snapshot.selected else " "
        sampled = " sampled" if candidate.token_id == snapshot.sampled_id else ""
        colour = YELLOW if index == snapshot.selected else GREY
        label = json.dumps(candidate.piece, ensure_ascii=False)
        bar = "█" * max(0, round(candidate.probability * 24))
        print(
            f"{colour}{cursor} {label:<22} {candidate.probability:9.4%} "
            f"{bar}{sampled}{RESET}"
        )
    auto_action = "pause" if auto else "play"
    print(
        f"{GREY}[←] undo  [→] commit/next  [↑/↓] select  "
        f"[a] {auto_action}  [q] cancel{RESET}",
        flush=True,
    )


def parse_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    args: dict[str, str] = {}
    for item in re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*:)", match.group("args")):
        key, separator, value = item.partition(":")
        if not separator:
            continue
        quoted = QUOTED_RE.fullmatch(value.strip())
        args[key.strip()] = quoted.group(1) if quoted else value.strip()
    return match.group("name"), args


class Explorer:
    def __init__(
        self,
        client: LlamaCpp,
        args: argparse.Namespace,
        messages: list[dict[str, Any]],
    ) -> None:
        self.client = client
        self.args = args
        self.auto = args.auto
        self.snapshots: list[Snapshot] = []
        self.index = 0
        self.seed_offset = 0
        self.phase = "first"
        self.messages = [message.copy() for message in messages]

        prompt = client.apply_template(self.messages)
        ids = client.tokenize(prompt)
        self.snapshots.append(Snapshot("request", ids, [], "INITIAL REQUEST"))

    @property
    def current(self) -> Snapshot:
        return self.snapshots[self.index]

    def generated_text(self, snapshot: Snapshot | None = None) -> str:
        return self.client.detokenize((snapshot or self.current).generated_ids)

    def trim_future(self) -> None:
        del self.snapshots[self.index + 1 :]

    def prepare_candidates(self) -> None:
        snapshot = self.current
        if snapshot.candidates or snapshot.kind not in {"request", "token"}:
            return
        sampled_id, candidates = self.client.candidates(
            snapshot.token_ids, self.args, self.args.seed + self.seed_offset
        )
        self.seed_offset += 1
        snapshot.sampled_id = sampled_id
        snapshot.candidates = candidates
        snapshot.selected = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.token_id == sampled_id
        )
        snapshot.label = "NEXT TOKEN"

    def commit_token(self) -> None:
        snapshot = self.current
        self.prepare_candidates()
        candidate = snapshot.candidates[snapshot.selected]
        self.trim_future()
        next_snapshot = Snapshot(
            "token",
            snapshot.token_ids + [candidate.token_id],
            snapshot.generated_ids + [candidate.token_id],
            "TOKEN COMMITTED",
            highlight_piece=candidate.piece,
        )
        self.snapshots.append(next_snapshot)
        self.index += 1

        text = self.generated_text(next_snapshot)
        if self.phase == "first" and "<tool_call|>" in text:
            self.auto = False
            self.add_tool_result(text)
        elif "<turn|>" in text:
            self.auto = False
            next_snapshot.kind = "done"
            next_snapshot.label = ""

    def add_tool_result(self, output: str) -> None:
        parsed = parse_tool_call(output)
        if not parsed:
            self.current.kind = "done"
            self.current.label = ""
            return
        name, arguments = parsed
        if name not in TOOL_IMPL:
            raise ServerError(f"Unsupported tool: {name}")
        result = TOOL_IMPL[name](**arguments)
        self.snapshots.append(
            Snapshot(
                "tool",
                self.current.token_ids.copy(),
                self.current.generated_ids.copy(),
                "TOOL RESULT",
                detail=f"{name}({json.dumps(arguments, ensure_ascii=False)})\n→ {result}",
            )
        )
        self.index += 1

    def start_new_request(self) -> None:
        output = self.generated_text()
        parsed = parse_tool_call(output)
        if not parsed:
            raise ServerError("No tool call available for the new request")
        name, arguments = parsed
        result = TOOL_IMPL[name](**arguments)
        self.messages.extend(
            [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "name": name, "content": result},
            ]
        )
        prompt = self.client.apply_template(self.messages)
        ids = self.client.tokenize(prompt)
        context = self.client.detokenize(ids)
        response_match = re.search(
            r"<\|tool_response>.*?<tool_response\|><\|channel>thought",
            context,
            re.DOTALL,
        )
        highlighted_response = response_match.group(0) if response_match else ""
        self.trim_future()
        self.snapshots.append(
            Snapshot(
                "request",
                ids,
                [],
                "NEW REQUEST",
                highlight_context=highlighted_response,
            )
        )
        self.index += 1
        self.phase = "second"

    def forward(self) -> None:
        snapshot = self.current
        if snapshot.kind == "tool":
            self.start_new_request()
        elif snapshot.kind == "done":
            return
        else:
            self.commit_token()

    def back(self) -> None:
        if self.index == 0:
            return
        self.index -= 1
        self.trim_future()
        snapshot = self.current
        if snapshot.kind == "tool":
            self.phase = "first"
        elif "<|tool_response>" in self.client.detokenize(snapshot.token_ids):
            self.phase = "second"
        else:
            self.phase = "first"
        self.auto = False

    def move_selection(self, delta: int) -> None:
        self.prepare_candidates()
        if self.current.candidates:
            self.current.selected = (self.current.selected + delta) % len(
                self.current.candidates
            )
            self.auto = False

    def run(self, screen: TerminalScreen) -> Snapshot | None:
        while True:
            self.prepare_candidates()
            screen.clear()
            render_snapshot(self.client, self.current, self.auto)
            if self.current.kind == "done":
                return self.current

            if not screen.interactive:
                self.forward()
                continue

            key = read_key(self.args.delay if self.auto else None)
            if key is None:
                self.forward()
            elif key == "q":
                return None
            elif key == "a":
                self.auto = not self.auto
            elif key == "left":
                self.back()
            elif key in {"right", "\r", "\n"}:
                self.forward()
            elif key == "up":
                self.move_selection(-1)
            elif key == "down":
                self.move_selection(1)


@dataclass
class TurnResult:
    raw_text: str
    prompt_tokens: int
    completion_tokens: int


class OpenCodeExplorer:
    """Generate one externally managed assistant turn."""

    def __init__(
        self,
        client: LlamaCpp,
        args: argparse.Namespace,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancelled: threading.Event,
    ) -> None:
        self.client = client
        self.args = args
        self.cancelled = cancelled
        self.auto = args.auto
        self.snapshots: list[Snapshot] = []
        self.index = 0
        self.seed_offset = 0

        prompt = client.apply_template(messages, tools)
        ids = client.tokenize(prompt)
        self.prompt_tokens = len(ids)
        self.snapshots.append(Snapshot("request", ids, [], "OPENCODE REQUEST"))

    @property
    def current(self) -> Snapshot:
        return self.snapshots[self.index]

    def prepare_candidates(self) -> None:
        snapshot = self.current
        if snapshot.candidates or snapshot.kind == "done":
            return
        sampled_id, candidates = self.client.candidates(
            snapshot.token_ids, self.args, self.args.seed + self.seed_offset
        )
        self.seed_offset += 1
        snapshot.sampled_id = sampled_id
        snapshot.candidates = candidates
        snapshot.selected = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.token_id == sampled_id
        )
        snapshot.label = "NEXT TOKEN"

    def commit_token(self) -> None:
        self.prepare_candidates()
        snapshot = self.current
        candidate = snapshot.candidates[snapshot.selected]
        del self.snapshots[self.index + 1 :]
        generated_ids = snapshot.generated_ids + [candidate.token_id]
        next_snapshot = Snapshot(
            "token",
            snapshot.token_ids + [candidate.token_id],
            generated_ids,
            "TOKEN COMMITTED",
            highlight_piece=candidate.piece,
        )
        self.snapshots.append(next_snapshot)
        self.index += 1
        generated = self.client.detokenize(generated_ids)
        if "<turn|>" in generated or len(generated_ids) >= self.args.max_tokens:
            next_snapshot.kind = "done"
            next_snapshot.label = ""
            self.auto = False

    def back(self) -> None:
        if self.index == 0:
            return
        self.index -= 1
        del self.snapshots[self.index + 1 :]
        self.auto = False

    def move_selection(self, delta: int) -> None:
        self.prepare_candidates()
        if self.current.candidates:
            self.current.selected = (self.current.selected + delta) % len(
                self.current.candidates
            )
            self.auto = False

    def run(self, screen: TerminalScreen) -> TurnResult:
        rendered_state: tuple[int, int, bool, str] | None = None
        while True:
            if self.cancelled.is_set():
                raise ServerError("OpenCode disconnected")
            self.prepare_candidates()
            state = (
                self.index,
                self.current.selected,
                self.auto,
                self.current.kind,
            )
            if state != rendered_state:
                render_persistent_snapshot(
                    self.client,
                    self.current,
                    self.auto,
                    show_context=rendered_state is None,
                )
                rendered_state = state
            if self.current.kind == "done":
                raw = self.client.detokenize(self.current.generated_ids)
                return TurnResult(raw, self.prompt_tokens, len(self.current.generated_ids))
            if not screen.interactive:
                self.commit_token()
                continue

            key = read_key(self.args.delay if self.auto else 0.25)
            if key is None:
                if self.auto:
                    self.commit_token()
                continue
            if key == "q":
                raise ServerError("Generation cancelled in Token Lens")
            if key == "a":
                self.auto = not self.auto
            elif key == "left":
                self.back()
            elif key in {"right", "\r", "\n"}:
                self.commit_token()
            elif key == "up":
                self.move_selection(-1)
            elif key == "down":
                self.move_selection(1)


class GemmaValueParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def skip_space(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def consume(self, value: str) -> None:
        self.skip_space()
        if not self.text.startswith(value, self.position):
            raise ValueError(f"Expected {value!r} at position {self.position}")
        self.position += len(value)

    def parse(self) -> Any:
        value = self.parse_value()
        self.skip_space()
        if self.position != len(self.text):
            raise ValueError(f"Unexpected input at position {self.position}")
        return value

    def parse_value(self) -> Any:
        self.skip_space()
        if self.text.startswith('<|"|>', self.position):
            return self.parse_string()
        if self.position < len(self.text) and self.text[self.position] == "{":
            return self.parse_object()
        if self.position < len(self.text) and self.text[self.position] == "[":
            return self.parse_array()
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in ",}]":
            self.position += 1
        token = self.text[start : self.position].strip()
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "null":
            return None
        try:
            return float(token) if any(char in token for char in ".eE") else int(token)
        except ValueError:
            return token

    def parse_string(self) -> str:
        self.consume('<|"|>')
        end = self.text.find('<|"|>', self.position)
        if end < 0:
            raise ValueError("Unterminated Gemma string")
        value = self.text[self.position : end]
        self.position = end + len('<|"|>')
        return value

    def parse_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self.consume("{")
        self.skip_space()
        if self.position < len(self.text) and self.text[self.position] == "}":
            self.position += 1
            return result
        while True:
            self.skip_space()
            start = self.position
            while self.position < len(self.text) and self.text[self.position] != ":":
                self.position += 1
            if self.position >= len(self.text):
                raise ValueError("Missing object key separator")
            key = self.text[start : self.position].strip()
            self.position += 1
            result[key] = self.parse_value()
            self.skip_space()
            if self.position < len(self.text) and self.text[self.position] == "}":
                self.position += 1
                return result
            self.consume(",")

    def parse_array(self) -> list[Any]:
        result: list[Any] = []
        self.consume("[")
        self.skip_space()
        if self.position < len(self.text) and self.text[self.position] == "]":
            self.position += 1
            return result
        while True:
            result.append(self.parse_value())
            self.skip_space()
            if self.position < len(self.text) and self.text[self.position] == "]":
                self.position += 1
                return result
            self.consume(",")


@dataclass
class ParsedAssistantTurn:
    content: str | None
    reasoning: str | None
    tool_calls: list[dict[str, Any]]


def parse_assistant_turn(raw: str) -> ParsedAssistantTurn:
    reasoning_blocks = re.findall(
        r"<\|channel>thought\s*\n?(.*?)<channel\|>", raw, re.DOTALL
    )
    calls: list[dict[str, Any]] = []
    call_pattern = re.compile(
        r"<\|tool_call>call:(?P<name>[^\s{]+)(?P<arguments>\{.*?\})<tool_call\|>",
        re.DOTALL,
    )
    for match in call_pattern.finditer(raw):
        arguments = GemmaValueParser(match.group("arguments")).parse()
        if not isinstance(arguments, dict):
            raise ServerError("Gemma emitted non-object tool arguments")
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": match.group("name"),
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )

    if calls:
        content = None
    else:
        content = raw.rsplit("<channel|>", 1)[-1]
        content = re.sub(r"<\|channel>[^\n]*\n?", "", content)
        content = content.replace("<turn|>", "").replace("<eos>", "").strip()
    reasoning = "\n\n".join(block.strip() for block in reasoning_blocks if block.strip())
    return ParsedAssistantTurn(content or None, reasoning or None, calls)


@dataclass
class CompletionJob:
    payload: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: TurnResult | None = None
    error: str | None = None


def completion_document(job: CompletionJob) -> dict[str, Any]:
    if job.result is None:
        raise ServerError(job.error or "Generation did not return a result")
    parsed = parse_assistant_turn(job.result.raw_text)
    created = int(time.time())
    model = str(job.payload.get("model", "gemma4:e2b"))
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
    if parsed.reasoning:
        message["reasoning"] = parsed.reasoning
    finish_reason = "stop"
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": job.result.prompt_tokens,
            "completion_tokens": job.result.completion_tokens,
            "total_tokens": job.result.prompt_tokens + job.result.completion_tokens,
        },
    }


class TokenLensHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], jobs: queue.Queue[CompletionJob]) -> None:
        super().__init__(address, TokenLensHandler)
        self.jobs = jobs


class TokenLensHandler(http.server.BaseHTTPRequestHandler):
    server: TokenLensHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "gemma4:e2b",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "token-lens",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload.get("messages"), list):
                raise ValueError("messages must be an array")
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        job = CompletionJob(payload)
        self.server.jobs.put(job)
        if payload.get("stream", False):
            self.stream_job(job)
        else:
            job.done.wait()
            if job.error:
                self.send_json(500, {"error": {"message": job.error, "type": "server_error"}})
            else:
                self.send_json(200, completion_document(job))

    def stream_job(self, job: CompletionJob) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while not job.done.wait(5):
                self.wfile.write(b": token-lens keepalive\n\n")
                self.wfile.flush()
            if job.error:
                event = {"error": {"message": job.error, "type": "server_error"}}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            document = completion_document(job)
            message = document["choices"][0]["message"]
            chunk = {
                "id": document["id"],
                "object": "chat.completion.chunk",
                "created": document["created"],
                "model": document["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": message,
                        "finish_reason": document["choices"][0]["finish_reason"],
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            if job.payload.get("stream_options", {}).get("include_usage"):
                usage = {
                    "id": document["id"],
                    "object": "chat.completion.chunk",
                    "created": document["created"],
                    "model": document["model"],
                    "choices": [],
                    "usage": document["usage"],
                }
                self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            job.cancelled.set()


def final_answer(text: str) -> str:
    if "<channel|>" in text:
        text = text.rsplit("<channel|>", 1)[1]
    return text.replace("<turn|>", "").strip()


def read_character() -> str:
    first = os.read(sys.stdin.fileno(), 1)
    if not first:
        return ""
    width = 1
    if first[0] & 0xE0 == 0xC0:
        width = 2
    elif first[0] & 0xF0 == 0xE0:
        width = 3
    elif first[0] & 0xF8 == 0xF0:
        width = 4
    data = first + (os.read(sys.stdin.fileno(), width - 1) if width > 1 else b"")
    return data.decode("utf-8", "ignore")


def chat_prompt(screen: TerminalScreen, default: str = "") -> str | None:
    if not sys.stdin.isatty():
        return default or None

    buffer = list(default)
    cursor = len(buffer)
    print(CURSOR_SHOW, end="", flush=True)

    def redraw() -> None:
        text = "".join(buffer)
        suffix = len(buffer) - cursor
        print(
            f"\r\033[2K{YELLOW}{BOLD}USER:{RESET} {text}",
            end="",
            flush=True,
        )
        if suffix:
            print(f"\033[{suffix}D", end="", flush=True)

    print()
    redraw()
    while True:
        char = read_character()
        if char in {"\r", "\n"}:
            print()
            print(CURSOR_HIDE, end="", flush=True)
            return "".join(buffer)
        if char == "\x04":
            print()
            print(CURSOR_HIDE, end="", flush=True)
            return None
        if char == "\x03":
            raise KeyboardInterrupt
        if char in {"\x7f", "\b"}:
            if cursor:
                cursor -= 1
                del buffer[cursor]
            redraw()
            continue
        if char == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.03)
            if ready and os.read(sys.stdin.fileno(), 1) == b"[":
                arrow = os.read(sys.stdin.fileno(), 1)
                if arrow == b"D" and cursor > 0:
                    cursor -= 1
                elif arrow == b"C" and cursor < len(buffer):
                    cursor += 1
                redraw()
            continue
        if char.isprintable():
            buffer.insert(cursor, char)
            cursor += 1
            redraw()


def run(args: argparse.Namespace) -> None:
    client = LlamaCpp(args.host)
    client.health()
    messages: list[dict[str, Any]] = []
    last_completed: tuple[Explorer, Snapshot] | None = None

    with TerminalScreen() as screen:
        prompt = chat_prompt(screen, args.prompt)
        while prompt is not None:
            prompt = prompt.strip()
            if prompt in {"/quit", "/exit"}:
                break
            if not prompt:
                prompt = chat_prompt(screen)
                continue

            messages.append({"role": "user", "content": prompt})
            explorer = Explorer(client, args, messages)
            completed = explorer.run(screen)
            if completed is None:
                break

            screen.clear()
            render_snapshot(client, completed, auto=False)
            last_completed = (explorer, completed)
            answer = final_answer(explorer.generated_text(completed))
            messages = explorer.messages + [{"role": "assistant", "content": answer}]

            if not sys.stdin.isatty():
                break
            prompt = chat_prompt(screen)

    if last_completed is not None:
        explorer, completed = last_completed
        render_snapshot(client, completed, auto=False)


def run_server(args: argparse.Namespace) -> None:
    client = LlamaCpp(args.host)
    client.health()
    jobs: queue.Queue[CompletionJob] = queue.Queue()
    server = TokenLensHTTPServer((args.listen, args.port), jobs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with TerminalScreen(alternate=False) as screen:
            heading("TOKEN LENS FOR OPENCODE")
            print(f"Listening on http://{args.listen}:{args.port}/v1")
            print(f"llama.cpp: {args.host}")
            request_number = 0
            while True:
                print(f"\n{GREY}Waiting for an OpenCode request …{RESET}", flush=True)
                job = jobs.get()
                request_number += 1
                heading(f"OPENCODE REQUEST {request_number}")
                print(
                    f"messages={len(job.payload.get('messages') or [])}  "
                    f"tools={len(job.payload.get('tools') or [])}  "
                    f"model={job.payload.get('model', '?')}"
                )
                if job.cancelled.is_set():
                    job.error = "Request was cancelled"
                    job.done.set()
                    continue
                try:
                    explorer = OpenCodeExplorer(
                        client,
                        args,
                        job.payload["messages"],
                        job.payload.get("tools") or [],
                        job.cancelled,
                    )
                    job.result = explorer.run(screen)
                except (ServerError, ValueError, KeyError) as exc:
                    job.error = str(exc)
                finally:
                    job.done.set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="http://127.0.0.1:8081")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--delay", type=float, default=0.04)
    parser.add_argument("--max-tokens", type=int, default=4096)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt", nargs="*", default=["Wie", "ist", "das", "Wetter", "in", "Berlin?"]
    )
    add_sampling_args(parser)
    args = parser.parse_args(argv)
    args.prompt = " ".join(args.prompt)
    args.mode = "demo"
    return args


def parse_server_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=f"{sys.argv[0]} serve")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    add_sampling_args(parser)
    args = parser.parse_args(argv)
    args.mode = "serve"
    return args


def main() -> int:
    args = (
        parse_server_args(sys.argv[2:])
        if len(sys.argv) > 1 and sys.argv[1] == "serve"
        else parse_args()
    )
    try:
        if args.mode == "serve":
            run_server(args)
        else:
            run(args)
    except ServerError as exc:
        print(f"\n{YELLOW}llama.cpp: {exc}{RESET}", file=sys.stderr)
        print(
            f"{GREY}Start llama-server on {args.host} with --special enabled.{RESET}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
