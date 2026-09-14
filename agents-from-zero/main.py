#!/usr/bin/env python3
"""
Bare-bones agent CLI.

    AGENT_BACKEND=claude python main.py     # hosted frontier model (Anthropic)
    AGENT_BACKEND=local  python main.py     # your own OpenAI-compatible server

Copy .env.example to .env, fill in what you need, then:

    pip install -r requirements.txt
    export $(grep -v '^#' .env | xargs)   # or use python-dotenv / direnv
    python main.py

See README.md for the concepts and for running an open-source model
locally (Ollama, llama.cpp) or via a hosted API.
"""
import os

from agent.core import Agent


def build_backend():
    kind = os.environ.get("AGENT_BACKEND", "claude").lower()

    if kind == "claude":
        from agent.llm_anthropic import AnthropicBackend

        # claude-haiku-4-5 is the cheapest current model and plenty for this
        # demo; swap in claude-sonnet-5 for tougher tasks. See:
        # https://platform.claude.com/docs/en/about-claude/models/overview
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return AnthropicBackend(model=model)

    if kind == "local":
        from agent.llm_openai_compat import OpenAICompatBackend

        model = os.environ.get("LOCAL_LLM_MODEL", "llama3.1")
        return OpenAICompatBackend(model=model)

    raise SystemExit(f"Unknown AGENT_BACKEND={kind!r}; use 'claude' or 'local'.")


def main():
    backend_kind = os.environ.get("AGENT_BACKEND", "claude")
    # Set AGENT_HISTORY_FILE="" to disable persistence and start fresh every run.
    history_file = os.environ.get("AGENT_HISTORY_FILE", "agent_history.json") or None
    agent = Agent(build_backend(), history_file=history_file)

    print(f"Bare-bones agent ready (backend={backend_kind}). Ctrl-D to quit.")
    if agent.history:
        print(f"Resumed {len(agent.history)} prior message(s) from {history_file}.")
    print()
    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        answer = agent.run(user_input)
        print(f"agent> {answer}\n")


if __name__ == "__main__":
    main()
