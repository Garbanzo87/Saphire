"""LLM providers.

* `MockLLM`      – deterministic, dependency-free scripted policy used for tests, CI and demos.
                    It behaves like a weak-but-plausible tool-using agent whose quality depends on
                    which tools are exposed to it (so tool routing / exemplars measurably matter).
* `LiteLLMProvider` – any hosted model (OpenAI, Anthropic, Azure, Bedrock, Ollama, vLLM ...) via LiteLLM.
* `HFLocalProvider` – a local HuggingFace checkpoint (e.g. one produced by `saphire train`).

`get_llm("openai/gpt-4o-mini")`, `get_llm("hf:./checkpoints/sft")`, `get_llm("mock")` pick the provider.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Optional, Protocol

from .types import LLMResponse, Message, Role, ToolCall, ToolSpec, Usage


class LLM(Protocol):
    model_name: str

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                 temperature: float = 0.0, **kw: Any) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Mock policy
# ---------------------------------------------------------------------------
_ID_PATTERNS = {
    "order_id": r"\bORD-\d+\b",
    "customer_id": r"\bCUS-\d+\b",
    "ticket_id": r"\bTKT-\d+\b",
    "invoice_id": r"\bINV-\d+\b",
    "product_id": r"\bSKU-\d+\b",
    "shipment_id": r"\bSHP-\d+\b",
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "amount": r"\$?(\d+(?:\.\d+)?)",
}
_STOP = set("the a an to of for and then please with on in at is it this that my me i you all".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


def _clauses(text: str) -> list[str]:
    parts = re.split(r"\bthen\b|\bafter that\b|;|\. |\? |\n|, and |\band then\b|\bfinally\b", text, flags=re.I)
    return [p.strip(" .,") for p in parts if p.strip(" .,")]


class MockLLM:
    """Scripted keyword policy.

    Given the conversation and the exposed tools it (1) splits the user's instructions into
    clauses, (2) for each clause not yet handled picks the exposed tool with the highest lexical
    overlap, (3) fills arguments from identifiers found in the *visible* context, and
    (4) answers when every clause has been handled.

    Parameters
    ----------
    error_rate: probability of confusing the best tool with the runner-up look-alike (no confusion when an
        in-context demonstration is available).
    context_window: only the last N messages are used to fill arguments (simulates context loss).
    seed: RNG seed for reproducibility.
    """

    model_name = "mock"

    def __init__(self, error_rate: float = 0.0, context_window: Optional[int] = None, seed: int = 0,
                 latency_ms: float = 0.0):
        self.error_rate = error_rate
        self.context_window = context_window
        self.rng = random.Random(seed)
        self.latency_ms = latency_ms

    # --- helpers ---
    def _visible(self, messages: list[Message]) -> list[Message]:
        if self.context_window is None:
            return messages
        sys_msgs = [m for m in messages if m.role == Role.system]
        rest = [m for m in messages if m.role != Role.system]
        return sys_msgs + rest[-self.context_window:]

    def _fill_args(self, spec: ToolSpec, clause: str, visible_text: str, tool_outputs: list[dict]) -> dict:
        args: dict[str, Any] = {}
        props = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        for pname, pschema in props.items():
            val: Any = None
            pat = _ID_PATTERNS.get(pname)
            # 1. identifiers mentioned in the current clause, then anywhere in the visible context
            if pat:
                m = re.search(pat, clause) or re.search(pat, visible_text)
                if m:
                    val = m.group(1) if m.groups() else m.group(0)
                    if pname == "amount":
                        val = float(val)
            # 2. enum-like / free-text parameters derived from the clause itself
            if val is None and pname == "status":
                m = re.search(r"\b(open|closed|resolved|pending|shipped|cancelled|escalated)\b", clause, flags=re.I)
                val = m.group(1).lower() if m else None
            if val is None and pname == "priority":
                m = re.search(r"\b(low|medium|high|urgent)\b", clause, flags=re.I)
                val = m.group(1).lower() if m else None
            if val is None and pname == "direction":
                m = re.search(r"\b(above|below)\b", clause, flags=re.I)
                val = m.group(1).lower() if m else None
            if val is None and pname in ("reason", "message", "query", "text", "note", "body", "question", "subject"):
                val = clause
            if val is None and pname in ("address", "new_address"):
                m = re.search(r"\bto\s+(.+)$", clause, flags=re.I)
                val = m.group(1) if m else clause
            if val is None and pname in ("metric", "table", "channel", "name", "format", "cron", "code"):
                pats = {"metric": r"\bmetric\s+([a-z_]+)", "table": r"\btable\s+([a-z_]+)", "channel": r"(#[a-z_-]+)",
                        "name": r"\bnamed\s+([a-z0-9_]+)", "format": r"\b(csv|pdf|json)\b", "cron": r"\bcron\s+([\d*/,\- ]+)$",
                        "code": r"\bcode\s+([A-Z0-9]+)"}
                m = re.search(pats[pname], clause, flags=re.I)
                val = m.group(1).strip() if m else None
            # 3. values carried from earlier tool outputs (context between steps)
            if val is None:
                for out in reversed(tool_outputs):
                    if isinstance(out, dict) and pname in out and out[pname] not in (None, ""):
                        val = out[pname]
                        break
            # 4. numeric fallbacks
            if val is None and pschema.get("type") in ("integer", "number"):
                m = re.search(r"\b(\d+(?:\.\d+)?)\b", clause)
                if m:
                    val = float(m.group(1)) if pschema.get("type") == "number" else int(float(m.group(1)))
            if val is None and pname in required:
                val = clause  # best effort
            if val is not None:
                args[pname] = val
        return args

    @staticmethod
    def _rules(messages: list[Message]) -> list[tuple[str, str]]:
        """Parse `Rule: when the request mentions "<phrase>", use <tool>.` lines from the system prompt."""
        rules = []
        for m in messages:
            if m.role == Role.system:
                rules += re.findall(r'Rule: when the request mentions "([^"]+)", use ([a-z_][a-z0-9_]*)', m.content)
        return rules

    @staticmethod
    def _exemplar_sequence(messages: list[Message]) -> list[str]:
        """Parse the first `Tools: a(...) -> b(...)` demonstration line from the system prompt."""
        for m in messages:
            if m.role == Role.system and "Tools:" in m.content:
                for line in m.content.splitlines():
                    line = line.strip()
                    if line.startswith("Tools:"):
                        return re.findall(r"([a-z_][a-z0-9_]*)\(", line[6:])
        return []

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                 temperature: float = 0.0, **kw: Any) -> LLMResponse:
        t0 = time.perf_counter()
        tools = tools or []
        visible = self._visible(messages)
        user_text = "\n".join(m.content for m in messages if m.role == Role.user)
        visible_text = "\n".join(m.content for m in visible if m.role in (Role.user, Role.tool))
        done_calls = [tc for m in messages if m.role == Role.assistant for tc in m.tool_calls]
        tool_outputs: list[dict] = []
        for m in visible:
            if m.role == Role.tool:
                try:
                    tool_outputs.append(json.loads(m.content))
                except Exception:
                    pass
        clauses = _clauses(user_text)
        handled = len(done_calls)
        response: Message
        if handled < len(clauses) and tools and handled < 12:
            clause = clauses[handled]
            ctoks = _tokens(clause)
            scored = []
            for pos, spec in enumerate(tools):
                ttoks = _tokens(spec.name.replace("_", " ") + " " + spec.description)
                overlap = len(ctoks & ttoks)
                # ties broken by position: tools listed first are preferred (as real LLMs do)
                scored.append((overlap, -pos, spec))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            best = scored[0][2]
            # explicit instructions in the system prompt (honoured when the named tool is exposed)
            by_name = {t.name: t for t in tools}
            rule_hit = None
            for phrase, tool_name in self._rules(messages):
                if phrase.lower() in clause.lower() and tool_name in by_name:
                    rule_hit = by_name[tool_name]
                    break
            if rule_hit is not None:
                best = rule_hit
            # in-context exemplars: follow a demonstrated tool sequence when the prompt provides one
            demo = self._exemplar_sequence(messages)
            if demo and handled < len(demo):
                by_name = {t.name: t for t in tools}
                if demo[handled] in by_name:
                    best = by_name[demo[handled]]
            if self.error_rate > 0 and not demo and rule_hit is None and len(scored) > 1 and self.rng.random() < self.error_rate:
                # confusion error: pick the runner-up candidate (a look-alike tool) instead of the best one
                best = scored[1][2]
            if scored[0][0] == 0 and not demo and self.rng.random() < 0.5:
                # nothing matches; skip this clause with a short text turn
                response = Message(role=Role.assistant, content=f"I could not find a tool for: {clause}")
            else:
                args = self._fill_args(best, clause, visible_text, tool_outputs)
                response = Message(role=Role.assistant, content="", tool_calls=[ToolCall(name=best.name, arguments=args)])
        else:
            summary = "; ".join(json.dumps(o, default=str)[:400] for o in tool_outputs[-3:])
            response = Message(role=Role.assistant, content=f"Done. {summary}".strip())
        ptoks = sum(len(m.content) // 4 + 8 for m in messages) + sum(len(json.dumps(t.to_openai())) // 4 for t in tools)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        return LLMResponse(message=response, usage=Usage(prompt_tokens=ptoks, completion_tokens=len(response.content) // 4 + 12),
                           model="mock", latency_ms=(time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# LiteLLM (hosted providers)
# ---------------------------------------------------------------------------
class LiteLLMProvider:
    def __init__(self, model: str, api_base: Optional[str] = None, max_tokens: int = 1024, **default_kw: Any):
        try:
            import litellm  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("Hosted providers require `pip install saphire[providers]`") from e
        self.model_name = model
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.default_kw = default_kw

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                 temperature: float = 0.0, **kw: Any) -> LLMResponse:
        import litellm

        t0 = time.perf_counter()
        params: dict[str, Any] = dict(model=self.model_name, messages=[m.to_openai() for m in messages],
                                      temperature=temperature, max_tokens=self.max_tokens, **self.default_kw, **kw)
        if tools:
            params["tools"] = [t.to_openai() for t in tools]
        if self.api_base:
            params["api_base"] = self.api_base
        resp = litellm.completion(**params)
        choice = resp.choices[0].message
        tcs = []
        for tc in getattr(choice, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            message=Message(role=Role.assistant, content=choice.content or "", tool_calls=tcs),
            usage=Usage(prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0),
            model=self.model_name, latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# Local HF checkpoint (text-format tool calling)
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def render_tools_text(tools: list[ToolSpec]) -> str:
    lines = ["You can call tools by emitting exactly one line of the form",
             '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>', "Available tools:"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description} args={json.dumps(t.parameters.get('properties', {}))}")
    return "\n".join(lines)


def messages_to_text(messages: list[Message], tools: list[ToolSpec] | None = None) -> str:
    """Plain-text chat serialisation used for SFT/DPO data and for HF local inference."""
    out = []
    for m in messages:
        if m.role == Role.system:
            content = m.content + ("\n\n" + render_tools_text(tools) if tools else "")
            out.append(f"<|system|>\n{content}")
        elif m.role == Role.user:
            out.append(f"<|user|>\n{m.content}")
        elif m.role == Role.assistant:
            out.append("<|assistant|>\n" + assistant_to_text(m))
        elif m.role == Role.tool:
            out.append(f"<|tool|>\n{m.content}")
    out.append("<|assistant|>\n")
    return "\n".join(out)


def assistant_to_text(m: Message) -> str:
    if m.tool_calls:
        tc = m.tool_calls[0]
        return f"<tool_call>{json.dumps({'name': tc.name, 'arguments': tc.arguments})}</tool_call>"
    return m.content


def parse_assistant_text(text: str) -> Message:
    m = TOOL_CALL_RE.search(text)
    if m:
        try:
            d = json.loads(m.group(1))
            return Message(role=Role.assistant, content="", tool_calls=[ToolCall(name=str(d.get("name", "")),
                                                                                  arguments=d.get("arguments") or {})])
        except json.JSONDecodeError:
            pass
    return Message(role=Role.assistant, content=text.strip())


class HFLocalProvider:
    def __init__(self, path: str, max_new_tokens: int = 96, device: str = "cpu"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Local HF models require `pip install saphire[train]`") from e
        self.model_name = f"hf:{path}"
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None,
                 temperature: float = 0.0, **kw: Any) -> LLMResponse:
        t0 = time.perf_counter()
        prompt = messages_to_text(messages, tools)
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-5), pad_token_id=self.tok.pad_token_id)
        gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.split("<|")[0]
        return LLMResponse(message=parse_assistant_text(gen),
                           usage=Usage(prompt_tokens=int(ids["input_ids"].shape[1]),
                                       completion_tokens=int(out.shape[1] - ids["input_ids"].shape[1])),
                           model=self.model_name, latency_ms=(time.perf_counter() - t0) * 1000)


def get_llm(model: str, **kw: Any) -> LLM:
    """Resolve a model string into a provider.

    - "mock", "mock:error=0.3,ctx=4,seed=1"
    - "hf:<path or hub id>"
    - anything else -> LiteLLM ("openai/gpt-4o-mini", "anthropic/claude-sonnet-4-20250514", "ollama/llama3", ...)
    """
    if model == "mock" or model.startswith("mock:"):
        opts: dict[str, Any] = {}
        if ":" in model:
            for kv in model.split(":", 1)[1].split(","):
                if "=" in kv:
                    k, v = kv.split("=")
                    opts[k.strip()] = v.strip()
        return MockLLM(error_rate=float(opts.get("error", 0.0)),
                       context_window=int(opts["ctx"]) if "ctx" in opts else None,
                       seed=int(opts.get("seed", 0)), latency_ms=float(opts.get("latency", 0.0)))
    if model.startswith("hf:"):
        return HFLocalProvider(model[3:], **kw)
    return LiteLLMProvider(model, **kw)
