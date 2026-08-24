"""
mitmproxy-Addon fuer die Workshop-Demo.

Zeigt den Verkehr zwischen einem Agent-Harness (z.B. OpenCode) und Ollama.

Start (Reverse-Proxy vor Ollama):
    mitmdump --mode reverse:http://localhost:11434 -p 11435 -s live-demo/oc_trace.py
"""

import json
from mitmproxy import http

BAR = "=" * 78

RESET = "\033[0m"
BOLD = "\033[1m"
WHITE = "\033[97m"
BLUE = "\033[94m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
GREY = "\033[90m"

# Konstante Felder der Stream-Blobs: einmal als Kopfzeile, danach ausgeblendet.
_CONSTANT_FIELDS = {"model", "id", "object", "created", "system_fingerprint", "index"}

# Bloecke im System-Prompt, die als Referenzmaterial gelten und zurueckgenommen
# dargestellt werden. Ein Tag zaehlt nur, wenn es allein auf der Zeile steht --
# sonst wuerde eine blosse Erwaehnung im Fliesstext den Block oeffnen.
_REFERENCE_BLOCKS = [
    ("<available_skills>", "</available_skills>"),
    ("<env>", "</env>"),
]


def _is_llm(flow):
    p = flow.request.path
    return "chat" in p or "generate" in p or "completions" in p


# ---------------------------------------------------------------------------
# REQUEST
# ---------------------------------------------------------------------------
def _content_lines(text):
    """Zeilen eines Message-Contents mit Markierung, ob sie Referenzmaterial sind."""
    in_ref = False
    closer = None
    for ln in text.split("\n"):
        stripped = ln.strip()
        if not in_ref:
            for opener, close in _REFERENCE_BLOCKS:
                if stripped == opener:
                    if opener == "<available_skills>":
                        yield "", False  # blank line before skills block
                    in_ref, closer = True, close
                    break
            yield ln, in_ref
        else:
            yield ln, True
            if stripped == closer:
                in_ref, closer = False, None
                if closer == "</env>":
                    yield "", False  # blank line after </env>


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(
                    p.get("text")
                    or p.get("content")
                    or json.dumps(p, ensure_ascii=False)
                )
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return "" if content is None else str(content)


def request(flow: http.HTTPFlow):
    if not _is_llm(flow):
        return
    try:
        obj = json.loads(flow.request.content or b"")
    except Exception:
        return

    print("\n" + BOLD + BAR + RESET)
    print(
        f"{BOLD}  HARNESS  ->  OLLAMA    {flow.request.method} {flow.request.path}{RESET}"
    )
    print(BOLD + BAR + RESET)
    print(
        f"{GREY}model={obj.get('model', '?')}  stream={obj.get('stream', False)}{RESET}"
    )

    msgs = obj.get("messages", [])
    print(f"\n\n{BOLD}{WHITE}MESSAGES ({len(msgs)}){RESET}")

    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        head = f"{BOLD}{BLUE}[{i}] {role.upper()}{RESET}"
        print(f"\n    {head}")

        text = _as_text(m.get("content"))
        if text:
            body = WHITE if role == "user" else ""
            for ln, is_ref in _content_lines(text):
                if is_ref:
                    print(f"        {GREY}| {ln}{RESET}")
                else:
                    print(f"        {GREY}|{RESET} {body}{ln}{RESET}")

        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            print(f"        {YELLOW}-> {fn.get('name')}({fn.get('arguments')}){RESET}")

        if role == "tool" and not text:
            print(f"        {GREY}(leer){RESET}")

    tools = obj.get("tools", [])
    if tools:
        print(f"\n\n{BOLD}{WHITE}TOOLS ({len(tools)}){RESET}")
        for i, t in enumerate(tools):
            fn = t.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            print(f"    {BOLD}{YELLOW}[{i}] {name}{RESET}")
            if desc:
                print(f"        {GREY}description: {desc}{RESET}")
            if params:
                print(
                    f"        {GREY}parameters: {json.dumps(params, ensure_ascii=False, indent=2)}{RESET}"
                )

    extra = {
        k: v
        for k, v in obj.items()
        if k not in {"model", "stream", "messages", "tools"}
    }
    if extra:
        parts = []
        for k, v in extra.items():
            sv = (
                v
                if isinstance(v, (int, float, bool, str))
                else json.dumps(v, ensure_ascii=False)
            )
            sv = str(sv)
            parts.append(f"{k}={sv[:60]}")
        print(f"{GREY}PARAMS: {'  '.join(parts)}{RESET}")


# ---------------------------------------------------------------------------
# RESPONSE
# ---------------------------------------------------------------------------
def _iter_chunks(raw: bytes):
    text = (raw or b"").decode("utf-8", "replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            line = line[6:]
        try:
            yield json.loads(line)
        except Exception:
            continue


def _flatten(obj):
    out = {}
    ch = obj.get("choices")
    if ch:
        delta = ch[0].get("delta") or {}
        fr = ch[0].get("finish_reason")
        if delta.get("reasoning"):
            out["thinking"] = delta["reasoning"]
        if delta.get("content"):
            out["content"] = delta["content"]
        if delta.get("tool_calls"):
            out["tool_calls"] = delta["tool_calls"]
        if fr:
            out["finish_reason"] = fr
    else:
        msg = obj.get("message") or {}
        if msg.get("thinking"):
            out["thinking"] = msg["thinking"]
        if msg.get("content"):
            out["content"] = msg["content"]
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        if obj.get("done"):
            out["done"] = True
            for k in (
                "done_reason",
                "eval_count",
                "prompt_eval_count",
                "eval_duration",
            ):
                if k in obj:
                    out[k] = obj[k]
    return out


def _render_field(field, value):
    if field == "thinking":
        return f"{GREY}thinking: {json.dumps(value, ensure_ascii=False)}{RESET}"
    if field == "content":
        return f"{WHITE}content: {json.dumps(value, ensure_ascii=False)}{RESET}"
    if field == "tool_calls":
        parts = []
        for c in value:
            fn = c.get("function", {})
            parts.append(f"{fn.get('name')}({fn.get('arguments')})")
        return f"{YELLOW}{BOLD}tool_calls: [{', '.join(parts)}]{RESET}"
    return f"{field}={json.dumps(value, ensure_ascii=False)}"


def response(flow: http.HTTPFlow):
    if not _is_llm(flow):
        return

    print("\n" + BOLD + BAR + RESET)
    print(f"{BOLD}  OLLAMA  ->  HARNESS{RESET}")
    print(BOLD + BAR + RESET)

    ct = flow.response.headers.get("content-type", "")
    raw = flow.response.content or b""
    streaming = "event-stream" in ct or "x-ndjson" in ct

    if not streaming:
        try:
            obj = json.loads(raw)
        except Exception:
            print(raw.decode("utf-8", "replace"))
            return
        for k, v in _flatten(obj).items():
            print("    " + _render_field(k, v))
        return

    chunks = list(_iter_chunks(raw))
    if not chunks:
        print(f"{GREY}(keine Chunks){RESET}")
        return

    # Accumulate full response
    accumulated = {
        "thinking": "",
        "content": "",
        "tool_calls": [],
        "finish_reason": None,
        "done": False,
    }
    meta_fields = {}

    for obj in chunks:
        flat = _flatten(obj)
        if not flat:
            continue

        for k, v in flat.items():
            if k in ("thinking", "content"):
                accumulated[k] += v
            elif k == "tool_calls":
                accumulated[k].extend(v)
            elif k in (
                "finish_reason",
                "done",
                "done_reason",
                "eval_count",
                "prompt_eval_count",
                "eval_duration",
            ):
                accumulated[k] = v
            else:
                meta_fields[k] = v

    # Show header once
    header = {k: v for k, v in chunks[0].items() if k in _CONSTANT_FIELDS}
    if header:
        pairs = "  ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in header.items()
        )
        print(f"\n{GREY}    in jedem Blob gleich:  {pairs}{RESET}")
        print(f"{GREY}    {'-' * 70}{RESET}")

    # Print accumulated response
    for field in ("thinking", "content", "tool_calls"):
        value = accumulated.get(field)
        if value:
            if field == "tool_calls" and value:
                print(f"\n    {_render_field(field, value)}")
            elif field in ("thinking", "content") and value.strip():
                print(f"\n    {_render_field(field, value)}")

    if accumulated.get("finish_reason"):
        print(f"\n    {GREY}finish_reason: {accumulated['finish_reason']}{RESET}")
    if accumulated.get("done"):
        meta_parts = []
        for k in ("done_reason", "eval_count", "prompt_eval_count", "eval_duration"):
            if k in accumulated:
                meta_parts.append(f"{k}={accumulated[k]}")
        if meta_parts:
            print(f"\n    {GREY}{'  '.join(meta_parts)}{RESET}")
