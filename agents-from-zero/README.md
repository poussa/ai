# agents-from-zero

A bare-bones AI agent, built with nothing but `requests` and the standard
library, so you can see every moving part instead of trusting a framework
to hide them. This is step one of learning to build agents from the ground
up; later steps (memory, planning, multi-agent, evals) all build on the
loop here.

## The idea, in one paragraph

An "agent" is a loop around a chat model plus some tools it can call:

1. Send the model a system prompt, the conversation so far, and a list of
   tools (name + description + JSON Schema for arguments).
2. The model replies either with a final answer, or with a request to call
   one of the tools (a "tool call" / "tool use").
3. If it's a tool call, you run the real Python function and hand the
   result back to the model as a new turn in the conversation.
4. Go to step 1. Repeat until the model gives a final answer or you hit a
   step limit (so a confused model can't loop forever).

That's the entire mechanism behind "agentic" behavior. Frameworks add
retries, memory, planning, and multi-agent orchestration on top, but the
loop itself is about 30 lines — see `agent/core.py`.

## Project layout

```
agent/
  tools.py             the toolbox: calculator, get_current_time, read_file
  llm_anthropic.py      backend adapter for the Claude API (frontier, hosted)
  llm_openai_compat.py  backend adapter for any OpenAI-compatible server
                         (Ollama, llama.cpp, LM Studio, vLLM, or a hosted
                         open-source-model API) — this is how you point the
                         same loop at a model that isn't Claude
  core.py               the agent loop itself
main.py                 CLI: pick a backend with $AGENT_BACKEND, then chat
.env.example             copy to .env and fill in what you need
```

Both backend adapters implement the same interface —
`chat(system_prompt, messages, tools) -> {"content": str, "tool_calls": [...]}`
— but each hand-converts to that provider's real wire format (Anthropic's
`tool_use`/`tool_result` content blocks vs. OpenAI-style `tool_calls`
arrays). Reading both side by side is the fastest way to learn what a
"tool call" actually looks like on the wire, since every framework you'll
use later is just automating this conversion.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set AGENT_BACKEND and the matching credentials/model
export $(grep -v '^#' .env | xargs)
python main.py
```

Try asking it something that needs a tool, e.g. `what's 37 * 84?` or
`what time is it?`, and watch the `-> tool call` / `<- tool result` lines
that `core.py` prints — that's the loop happening in real time.

## Three ways to run it

The agent loop doesn't care which model is behind it, as long as the
backend adapter speaks that provider's wire format. `AGENT_BACKEND`
picks the adapter; everything else is just which model you point it at.

### 1. A frontier model (Claude, hosted)

Simplest path, and a good place to start even before touching local
hardware, because it isolates the agent-loop concepts from GPU/driver
debugging.

1. Get a key at <https://console.anthropic.com/>.
2. In `.env`: `AGENT_BACKEND=claude`, `ANTHROPIC_API_KEY=sk-ant-...`.
3. Default model is `claude-haiku-4-5-20251001` — the cheapest current
   model, plenty for this demo (a handful of cents for a long session).
   Swap `ANTHROPIC_MODEL` to `claude-sonnet-5` if you want a stronger
   model for harder tasks. Current model IDs and pricing:
   <https://platform.claude.com/docs/en/about-claude/models/overview>

### 2. Fully local, on your own machine (Ollama or llama.cpp)

Set `AGENT_BACKEND=local` and point it at a server running on your own
hardware — no API key, no data leaving your machine. Any server that
exposes an OpenAI-compatible `/v1/chat/completions` endpoint works;
`agent/llm_openai_compat.py` doesn't know or care that it's local.

- **Ollama** — simplest CLI setup, works on Mac/Linux/Windows:
  ```bash
  brew install ollama          # or see https://ollama.com/download
  ollama serve                 # exposes an OpenAI-compatible API
  ollama pull llama3.2         # or qwen2.5:7b, mistral-nemo, etc.
  ```
  Then in `.env`:
  ```bash
  AGENT_BACKEND=local
  LOCAL_LLM_BASE_URL=http://localhost:11434/v1
  LOCAL_LLM_MODEL=llama3.2
  ```

- **llama.cpp** — the reference C++ inference engine; its built-in server
  also speaks the OpenAI-compatible API:
  ```bash
  brew install llama.cpp       # or build from source
  llama-server -hf <org>/<model-gguf> --port 8080
  ```
  Then in `.env`:
  ```bash
  AGENT_BACKEND=local
  LOCAL_LLM_BASE_URL=http://localhost:8080/v1
  LOCAL_LLM_MODEL=<whatever name the server reports>
  ```

- **LM Studio** — a free GUI that uses Apple's **MLX** runtime on Apple
  Silicon (generally the fastest option on M-series chips, since it's
  built for the unified-memory architecture rather than adapted from a
  CUDA/Vulkan codebase). Its one-click "start local server" button serves
  the same OpenAI-compatible API on `http://localhost:1234/v1`. Good if
  you'd rather not use the command line.

**Worked example — Mac M1, 16GB RAM:** that's enough unified memory for a
7B–8B model at Q4 quantization with a reasonable context window (13B is
possible but tight). Generation is workable for learning and
single-tool-call turns, but noticeably slower than a hosted frontier model
once the loop needs several rounds of tool calls in a row. As a rough
guide for any machine: 8GB is comfortable for quantized 3B models (7B at
Q4 is possible but leaves little headroom); 16GB is comfortable for 7B–8B;
24GB+ opens up 13B–34B.

### 3. An open-source model via a hosted API (e.g. Llama 3.2)

Don't have the hardware for local inference, but still want to try an
open-source model instead of Claude? Point the same `local` backend at a
hosted provider that serves open-source models over an OpenAI-compatible
API (Together AI, Groq, Fireworks, Hugging Face Inference Endpoints, and
others all qualify). In `.env`:

```bash
AGENT_BACKEND=local
LOCAL_LLM_BASE_URL=<the provider's base URL, ending in /v1>
LOCAL_LLM_MODEL=<the exact model name/id the provider lists>
LOCAL_LLM_API_KEY=<the API key the provider gave you>
```

This is the same code path as running fully local — the adapter only
speaks HTTP to a `/v1/chat/completions` endpoint, so it doesn't matter
whether that endpoint is a process on your laptop or someone else's GPU.

Tool-calling quality depends heavily on which model you pick, wherever
it's hosted — use one whose model card explicitly advertises tool/function
calling support (Qwen2.5/3.x-Instruct, Llama 3.1/3.3-Instruct,
Mistral-Nemo). Small heavily-quantized models frequently emit malformed
tool-call JSON; seeing that failure mode first-hand is itself a useful
lesson in why real agent frameworks add repair/retry logic around this
exact step.

## Where to go next

Once this loop feels obvious, natural extensions to try (each is a small,
self-contained upgrade to this same codebase):

- Persist `agent.history` to disk so conversations survive a restart.
- Add a tool that calls a real API (weather, search) to see how the model
  handles a tool that can fail or time out.
- Cap tool output size / add a tool allow-list to think about what a
  malicious or buggy tool result could do to the agent (prompt injection
  via tool results is a real, current attack class).
- Try the same prompts against multiple backends and compare how reliably
  each one produces well-formed tool calls.
