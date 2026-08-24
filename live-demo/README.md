# Live demos

These demos expose two parts of an agent run that chat interfaces usually hide:

- `oc_trace.py` prints the HTTP traffic between OpenCode and Ollama.
- `token_lens_llamacpp.py` generates one token per llama.cpp request and shows the model context, token candidates, tool call, and tool result.

Run all commands from the repository root.

## Requirements

- Python 3.10 or newer and [uv](https://docs.astral.sh/uv/)
- An ANSI-compatible terminal
- [Ollama](https://ollama.com/) and [mitmproxy](https://mitmproxy.org/) for the HTTP trace
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the token demo

`token_lens_llamacpp.py` declares its Python dependency inline, so `uv run` installs `requests` automatically.

## Trace OpenCode and Ollama

`oc_trace.py` is a mitmproxy addon. It prints messages, tool definitions, tool calls, tool results, and selected request parameters. It supports Ollama's native chat endpoint and the OpenAI-compatible `/v1/chat/completions` endpoint used by OpenCode.

### 1. Start Ollama

Make sure Ollama is available at `http://localhost:11434` and that the model configured in OpenCode is installed.

### 2. Start the reverse proxy

```bash
mitmdump --mode reverse:http://localhost:11434 -p 11435 -s live-demo/oc_trace.py
```

mitmproxy now listens on port `11435` and forwards requests to Ollama on port `11434`.

### 3. Connect OpenCode to the proxy

Copy `opencode.json.example` from this repository to `~/.config/opencode/` and remove `.example` from the filename. Then change the Ollama provider's `baseURL` in `opencode.json`:

```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama via mitmproxy",
      "options": {
        "baseURL": "http://localhost:11435/v1"
      },
      "models": {
        "gemma4:e2b": {
          "name": "Gemma 4 e2b"
        }
      }
    }
  }
}
```

Review the remaining providers, placeholders, and permissions in `opencode.json` before starting OpenCode.

### 4. Trigger a tool call

Start OpenCode, select the configured Ollama model, and ask it to read a file. The proxy terminal shows the request from OpenCode and the model's response.

The trace makes the division of work explicit. The model emits a structured tool call. OpenCode executes the tool, adds its result to the message history, and sends another request to the model.

### Reset the configuration

After the demo, change the `baseURL` in `~/.config/opencode/opencode.json` back to `http://localhost:11434/v1`.

## Generate one token at a time

`token_lens_llamacpp.py` uses Gemma 4's embedded chat template and requests exactly one token from llama.cpp at a time. It keeps the growing sequence as token IDs instead of decoding and tokenizing generated text again.

The demo includes a local `get_weather` tool. Once the model emits a complete tool call, the script runs the Python function, inserts the result through the chat template, and continues generation.

### 1. Download the model

```bash
mkdir -p ~/.cache/llama.cpp
curl -L \
  'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf?download=true' \
  -o ~/.cache/llama.cpp/gemma-4-E2B-it-Q4_K_M.gguf
```

Tested file:

```text
size:   3,106,738,272 bytes
sha256: 740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8
```

The GGUF contains the tokenizer and chat template. This text-only demo does not need a multimodal projector.

### 2. Start llama.cpp

```bash
llama-server \
  -m ~/.cache/llama.cpp/gemma-4-E2B-it-Q4_K_M.gguf \
  -c 4096 \
  -ngl all \
  --special \
  --no-mmproj \
  -np 1 \
  --host 127.0.0.1 \
  --port 8081 \
  --no-ui
```

The demo was tested with Homebrew llama.cpp build 10450. The script uses `/apply-template`, `/tokenize`, `/detokenize`, and `/completion`.

### 3. Run the demo

```bash
uv run live-demo/token_lens_llamacpp.py "Wie ist das Wetter in Berlin?"
```

The initial question remains editable until you press Enter. The terminal then shows two views:

- `RUN HISTORY` contains everything observed during the run, including reasoning removed from later model requests.
- `CURRENT MODEL CONTEXT` is decoded from the exact token IDs sent with the next `/completion` request.

Controls:

```text
Enter  generate the next token
a      continue to the end of the current model message
q      stop the demo
```

Automatic playback stops after each model message. Press `a` again to continue the next message. After a completed answer, enter another user message to continue the conversation. Use `/quit`, `/exit`, or `Ctrl-D` to leave.

Useful options:

```bash
uv run live-demo/token_lens_llamacpp.py --auto --delay 0.04 --top 5
```

Use `--host` to set another llama.cpp address. `--temperature`, `--top-k`, `--top-p`, and `--seed` control sampling. `--max-tokens` limits one model message.

Ghostty users may need to disable programming ligatures so special tokens remain readable and copyable:

```ini
font-feature = -calt, -liga, -dlig
```

## Connect the token demo to OpenCode

The script can also expose an OpenAI-compatible endpoint on port `8082`:

```bash
uv run live-demo/token_lens_llamacpp.py serve
```

Configure an OpenCode provider with the `baseURL` `http://127.0.0.1:8082/v1`. The llama.cpp server must remain available on port `8081`. Each OpenCode request then appears in the token interface before the script returns the completed response.

This mode is intended for the live demo, not as a general-purpose inference server. It processes the visible generation jobs one at a time.
