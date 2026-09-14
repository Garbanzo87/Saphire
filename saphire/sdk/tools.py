"""Tool registry: expose Python callables, MCP servers, SQL tables and HTTP APIs as tools.

A `ToolRegistry` is the "tools" half of a company's environment. It turns arbitrary
callables into OpenAI-function-calling JSON schemas so any LLM (or the mock policy)
can select and call them, and it records latency/errors for evaluation.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
import typing as t
from dataclasses import dataclass

from .types import ToolCall, ToolResult, ToolSpec

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema_for(fn: t.Callable) -> dict:
    sig = inspect.signature(fn)
    hints = t.get_type_hints(fn) if hasattr(fn, "__annotations__") else {}
    props: dict[str, dict] = {}
    required: list[str] = []
    doc_params = _parse_param_docs(fn.__doc__ or "")
    for name, p in sig.parameters.items():
        if name in ("self", "state", "ctx"):
            continue
        typ = hints.get(name, str)
        origin = t.get_origin(typ)
        if origin is t.Union or str(origin) == "types.UnionType":
            args = [a for a in t.get_args(typ) if a is not type(None)]
            typ = args[0] if args else str
        jtype = _PY_TO_JSON.get(typ, "string")
        entry: dict = {"type": jtype}
        if name in doc_params:
            entry["description"] = doc_params[name]
        props[name] = entry
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _parse_param_docs(doc: str) -> dict[str, str]:
    """Parse `name: description` lines under an `Args:` header (Google style)."""
    out: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        s = line.strip()
        if s.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if not s or s.endswith(":") and " " not in s:
                in_args = False
                continue
            if ":" in s:
                k, v = s.split(":", 1)
                out[k.strip().split(" ")[0]] = v.strip()
    return out


@dataclass
class RegisteredTool:
    spec: ToolSpec
    fn: t.Callable
    takes_state: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    # ---------- registration ----------
    def register(self, fn: t.Callable | None = None, *, name: str | None = None, description: str | None = None,
                 tags: list[str] | None = None, source: str = "python"):
        """Decorator / function to register a Python callable as a tool.

        If the callable's first parameter is named `state`, the environment state object
        is injected at call time (so tools can mutate a simulated world).
        """

        def _do(f: t.Callable):
            nm = name or f.__name__
            desc = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()
            spec = ToolSpec(name=nm, description=desc, parameters=_schema_for(f), tags=tags or [], source=source)
            params = list(inspect.signature(f).parameters)
            self._tools[nm] = RegisteredTool(spec=spec, fn=f, takes_state=bool(params) and params[0] == "state")
            return f

        return _do(fn) if fn is not None else _do

    def add_spec(self, spec: ToolSpec, fn: t.Callable) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, fn=fn)

    def merge(self, other: "ToolRegistry") -> "ToolRegistry":
        self._tools.update(other._tools)
        return self

    # ---------- lookup ----------
    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, names: t.Iterable[str] | None = None) -> list[ToolSpec]:
        if names is None:
            return [rt.spec for rt in self._tools.values()]
        return [self._tools[n].spec for n in names if n in self._tools]

    def get(self, name: str) -> RegisteredTool:
        return self._tools[name]

    # ---------- execution ----------
    def call(self, call: ToolCall, state: t.Any = None) -> ToolResult:
        t0 = time.perf_counter()
        rt = self._tools.get(call.name)
        if rt is None:
            return ToolResult(call_id=call.id, name=call.name, error=f"unknown tool '{call.name}'",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        try:
            args = dict(call.arguments or {})
            if rt.takes_state:
                out = rt.fn(state, **args)
            else:
                out = rt.fn(**args)
            if inspect.isawaitable(out):
                out = _run_sync(out)
            # ensure JSON-serialisable
            json.dumps(out, default=str)
            return ToolResult(call_id=call.id, name=call.name, output=out, latency_ms=(time.perf_counter() - t0) * 1000)
        except TypeError as e:  # bad arguments
            return ToolResult(call_id=call.id, name=call.name, error=f"invalid arguments: {e}",
                              latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # tool raised
            return ToolResult(call_id=call.id, name=call.name, error=f"{type(e).__name__}: {e}",
                              latency_ms=(time.perf_counter() - t0) * 1000)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # already inside an event loop: run in a fresh thread
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


# ---------------------------------------------------------------------------
# MCP adapter (optional dependency)
# ---------------------------------------------------------------------------
def registry_from_mcp(command: str, args: list[str] | None = None, env: dict[str, str] | None = None,
                      tags: list[str] | None = None) -> ToolRegistry:
    """Connect to a Model Context Protocol server over stdio and expose its tools.

    Each tool call opens a short-lived session (simple and robust; fine for evaluation
    workloads). Requires `pip install saphire[mcp]`.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:  # pragma: no cover
        raise ImportError("MCP support requires `pip install saphire[mcp]`") from e

    params = StdioServerParameters(command=command, args=args or [], env=env)

    async def _list():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return (await s.list_tools()).tools

    async def _call(name: str, arguments: dict):
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(name, arguments)
                parts = []
                for c in res.content:
                    parts.append(getattr(c, "text", None) or c.model_dump())
                return parts[0] if len(parts) == 1 else parts

    reg = ToolRegistry()
    for tool in _run_sync(_list()):
        spec = ToolSpec(name=tool.name, description=tool.description or "",
                        parameters=tool.inputSchema or {"type": "object", "properties": {}},
                        tags=tags or ["mcp"], source="mcp")

        def _make(nm):
            def _fn(**kwargs):
                return _run_sync(_call(nm, kwargs))

            return _fn

        reg.add_spec(spec, _make(tool.name))
    return reg
