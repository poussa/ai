"""
Minimal Claude (Anthropic Messages API) backend -- raw HTTPS, no SDK.

This talks directly to https://api.anthropic.com/v1/messages so you can see
exactly what an "agent framework" sends under the hood: a system prompt, a
list of messages, and a list of tool schemas. Anthropic's own Python SDK
(pip install anthropic) wraps this same call; using requests instead keeps
every byte on the wire visible for learning purposes.

Docs: https://platform.claude.com/docs/en/api/messages
Model IDs: https://platform.claude.com/docs/en/about-claude/models/overview
"""
import os

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicBackend:
    def __init__(self, model: str, api_key: str = None, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Get a key at "
                "https://console.anthropic.com/ and put it in your .env "
                "(see .env.example)."
            )
        self.max_tokens = max_tokens

    def chat(self, system_prompt: str, messages: list, tools: list) -> dict:
        """
        messages: canonical history -- a list of turns shaped like:
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "...", "tool_calls": [...]}
            {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
        tools: canonical schemas: [{"name", "description", "parameters"}]

        Returns a normalized assistant turn:
            {"content": str, "tool_calls": [{"id", "name", "arguments"}]}
        """
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": self._to_wire_messages(messages),
        }
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]

        resp = requests.post(
            API_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text}")
        return self._from_wire_response(resp.json())

    @staticmethod
    def _to_wire_messages(messages: list) -> list:
        wire = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})

            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                    )
                wire.append({"role": "assistant", "content": blocks})

            elif m["role"] == "tool":
                # Anthropic has no "tool" role -- a tool result is sent back
                # as a user message containing a tool_result content block.
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                        ],
                    }
                )
        return wire

    @staticmethod
    def _from_wire_response(data: dict) -> dict:
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({"id": block["id"], "name": block["name"], "arguments": block.get("input", {})})
        return {"content": content_text, "tool_calls": tool_calls}
