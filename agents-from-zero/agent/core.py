"""
The agent loop -- the part every framework (LangChain, etc.) wraps up and
hides behind an `.invoke()`. Stripped to its essence, an agent is:

    1. Give the model a goal, a system prompt, and a list of tools.
    2. Ask the model what to do next.
    3. If it wants to call a tool, run the tool and hand back the result.
    4. If it gives a final answer instead, stop.
    5. Repeat, with a hard cap so a confused model can't loop forever.

That's the whole idea. Everything else (memory, planning, multi-agent,
retries, guardrails) is refinement on top of this loop.
"""
from . import tools as tool_registry

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use a tool when it gives you information you don't already have or can't
compute reliably yourself (arithmetic, the current time, file contents).
Once you have enough information, answer the user directly in plain text
without calling a tool."""


class Agent:
    def __init__(self, backend, system_prompt: str = SYSTEM_PROMPT, max_steps: int = 6, verbose: bool = True):
        self.backend = backend
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []  # canonical message list; persists across .run() calls

    def _log(self, *parts):
        if self.verbose:
            print(*parts)

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        tool_schemas = tool_registry.get_tool_schemas()

        for _ in range(self.max_steps):
            reply = self.backend.chat(self.system_prompt, self.history, tool_schemas)

            if reply["tool_calls"]:
                self.history.append(
                    {"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}
                )
                for call in reply["tool_calls"]:
                    self._log(f"  -> {call['name']}({call['arguments']})")
                    result = tool_registry.call_tool(call["name"], call["arguments"])
                    self._log(f"  <- {result}")
                    self.history.append(
                        {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result}
                    )
                continue  # feed the tool result back and let the model decide what's next

            # No tool call: the model is done, and this is its final answer.
            self.history.append({"role": "assistant", "content": reply["content"]})
            return reply["content"]

        return "(gave up: hit max_steps without reaching a final answer)"
