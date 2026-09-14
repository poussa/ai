"""
OpenAI-compatible chat-completions backend -- raw HTTPS, no SDK.

This is the backend you point at a model running on your own machine.
Ollama, llama.cpp's built-in server, LM Studio, and vLLM all expose the
same /v1/chat/completions shape, so this one adapter works for any of
them -- just change LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL.

Whether tool-calling actually works well depends on the model, not just
the server: pick a model card that advertises "tools" / "function calling"
support (e.g. Qwen2.5, Llama 3.1/3.3, Mistral-Nemo). Small quantized
models frequently emit malformed tool calls -- that failure mode is part
of the lesson.

Ollama's OpenAI-compatible endpoint: http://localhost:11434/v1
"""
import json
import os

import requests


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 1024):
        self.model = model
        self.base_url = (base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        # Most local servers ignore the key entirely; the header slot exists
        # for anything (e.g. a reverse proxy) that fronts them with auth.
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
        self.max_tokens = max_tokens

    def chat(self, system_prompt: str, messages: list, tools: list) -> dict:
        wire_messages = [{"role": "system", "content": system_prompt}] + self._to_wire_messages(messages)

        payload = {"model": self.model, "messages": wire_messages, "max_tokens": self.max_tokens}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools
            ]

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            json=payload,
            timeout=180,  # local inference on modest hardware can be slow
        )
        if not resp.ok:
            raise RuntimeError(f"Local LLM server error {resp.status_code}: {resp.text}")
        return self._from_wire_response(resp.json())

    @staticmethod
    def _to_wire_messages(messages: list) -> list:
        wire = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})

            elif m["role"] == "assistant":
                entry = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                wire.append(entry)

            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return wire

    @staticmethod
    def _from_wire_response(data: dict) -> dict:
        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": args})
        return {"content": content, "tool_calls": tool_calls}
