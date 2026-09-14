"""
The agent's toolbox.

Each tool is plain Python: a name, a description the model reads to decide
when to use it, a JSON Schema for its arguments, and a function that
actually does the work. Nothing here depends on any agent framework --
this *is* the whole "function calling" / "tool use" mechanism, spelled out.

Add a new tool by writing a function and adding one entry to TOOLS below.
"""
import ast
import datetime
import operator
import os

# --- tool implementations ---------------------------------------------------


def _safe_eval(expr: str):
    """Evaluate a numeric expression without running eval() on arbitrary code.

    Only the AST nodes below are allowed, so 'expression' can never do
    anything except arithmetic -- unlike a bare eval(expr), which would let
    a misbehaving model (or a prompt-injected tool result) run any Python.
    """
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numbers are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def calculator(expression: str) -> str:
    try:
        return str(_safe_eval(expression))
    except Exception as e:
        return f"error: could not evaluate '{expression}': {e}"


def get_current_time() -> str:
    # Deliberately bare-bones: real timezone handling would use zoneinfo.
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_file(path: str) -> str:
    try:
        full_path = os.path.abspath(os.path.expanduser(path))
        with open(full_path, "r", errors="replace") as f:
            content = f.read()
        limit = 4000
        if len(content) > limit:
            return content[:limit] + f"\n...[truncated, {len(content)} chars total]"
        return content
    except Exception as e:
        return f"error: could not read '{path}': {e}"


# --- registry ----------------------------------------------------------------
#
# "schema" is what gets sent to the model (name, description, JSON Schema for
# arguments) -- this is the ONLY information the model has to decide when and
# how to call the tool. "function" is what actually runs once the model asks.

TOOLS = {
    "calculator": {
        "schema": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A numeric expression using + - * / ** ( ).",
                    }
                },
                "required": ["expression"],
            },
        },
        "function": calculator,
    },
    "get_current_time": {
        "schema": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": get_current_time,
    },
    "read_file": {
        "schema": {
            "name": "read_file",
            "description": "Read a small text file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."}
                },
                "required": ["path"],
            },
        },
        "function": read_file,
    },
}


def get_tool_schemas():
    """The list of tool schemas to hand to the model this turn."""
    return [t["schema"] for t in TOOLS.values()]


def call_tool(name: str, arguments: dict) -> str:
    """Run a tool by name and always return a string (tool results go back
    into the conversation as text, whatever backend is in use)."""
    if name not in TOOLS:
        return f"error: unknown tool '{name}'"
    try:
        return str(TOOLS[name]["function"](**arguments))
    except Exception as e:
        return f"error: tool '{name}' raised: {e}"
