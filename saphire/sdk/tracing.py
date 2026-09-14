"""OpenTelemetry-based tracing for agents.

    from saphire.sdk import tracing
    tracing.init(service_name="support-agent")            # exports to $SAPHIRE_HOST
    @tracing.trace(name="handle_ticket")
    def handle(ticket): ...
    with tracing.span("llm", kind="llm", input=..., output=...): ...

Spans are exported with the OTLP-compatible OpenInference-style attribute names
(`saphire.kind`, `input.value`, `output.value`, `llm.token_count.*`, `tool.name`), so
third-party OpenInference/OTel instrumentors (OpenAI, Anthropic, LangChain, ...) can be
attached to the same tracer provider and land in the same Saphire project.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import threading
import time
import typing as t
from typing import Any, Optional

import httpx
from opentelemetry import trace as ot
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult

_state: dict[str, Any] = {"provider": None, "exporter": None, "service": "saphire-agent"}
_local = threading.local()


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent = span.parent.span_id if span.parent else None
    attrs = dict(span.attributes or {})
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent, "016x") if parent else None,
        "name": span.name,
        "kind": attrs.get("saphire.kind", "span"),
        "start_time": span.start_time / 1e9 if span.start_time else None,
        "end_time": span.end_time / 1e9 if span.end_time else None,
        "status": span.status.status_code.name.lower() if span.status else "unset",
        "attributes": {k: v for k, v in attrs.items() if k != "saphire.kind"},
        "service": (span.resource.attributes.get("service.name") if span.resource else None),
        "events": [{"name": e.name, "time": e.timestamp / 1e9, "attributes": dict(e.attributes or {})} for e in span.events],
    }


class SaphireSpanExporter(SpanExporter):
    """Posts spans as JSON to the Saphire server (`POST /v1/traces/ingest`)."""

    def __init__(self, host: str, api_key: str, project: str, timeout: float = 10.0):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.project = project
        self.client = httpx.Client(timeout=timeout)
        self.failed = 0

    def export(self, spans: t.Sequence[ReadableSpan]) -> SpanExportResult:
        payload = {"project": self.project, "spans": [span_to_dict(s) for s in spans]}
        try:
            r = self.client.post(f"{self.host}/v1/traces/ingest", json=payload,
                                 headers={"x-api-key": self.api_key})
            if r.status_code >= 300:
                self.failed += 1
                return SpanExportResult.FAILURE
            return SpanExportResult.SUCCESS
        except Exception:
            self.failed += 1
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.client.close()


class InMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def export(self, spans: t.Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(span_to_dict(s) for s in spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


def init(service_name: str = "saphire-agent", host: Optional[str] = None, api_key: Optional[str] = None,
         project: Optional[str] = None, exporter: Optional[SpanExporter] = None, batch: bool = True,
         force: bool = False) -> TracerProvider:
    """Initialise the global tracer provider.

    Exports to `host` (default `$SAPHIRE_HOST` or http://localhost:8000) unless an explicit
    `exporter` is passed. Set `SAPHIRE_TRACING=off` to disable exporting entirely.
    """
    if _state["provider"] is not None and not force:
        return _state["provider"]
    host = host or os.getenv("SAPHIRE_HOST", "http://localhost:8000")
    api_key = api_key or os.getenv("SAPHIRE_API_KEY", "dev-key")
    project = project or os.getenv("SAPHIRE_PROJECT", "default")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is None and os.getenv("SAPHIRE_TRACING", "on").lower() not in ("off", "0", "false"):
        exporter = SaphireSpanExporter(host, api_key, project)
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter))
    old = _state["provider"]
    if old is not None:
        old.shutdown()
    if not _state.get("global_set"):
        # OpenTelemetry only allows the global provider to be set once; third-party instrumentors
        # (OpenInference etc.) attach to it. Saphire's own spans always use _state["provider"].
        ot.set_tracer_provider(provider)
        _state["global_set"] = True
    _state.update(provider=provider, exporter=exporter, service=service_name)
    return provider


def tracer():
    if _state["provider"] is None:
        init(exporter=InMemorySpanExporter(), batch=False)
    return _state["provider"].get_tracer("saphire")


def _attr(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, default=str)[:20000]
    except Exception:
        return str(v)[:20000]


@contextlib.contextmanager
def span(name: str, kind: str = "span", input: Any = None, output: Any = None, **attrs: Any):
    """Context manager creating a span. `kind` ∈ {agent, llm, tool, retriever, step, eval, span}."""
    with tracer().start_as_current_span(name) as s:
        s.set_attribute("saphire.kind", kind)
        if input is not None:
            s.set_attribute("input.value", _attr(input))
        for k, v in attrs.items():
            s.set_attribute(k, _attr(v))
        holder = _SpanHandle(s)
        try:
            yield holder
        except Exception as e:
            s.record_exception(e)
            s.set_status(ot.Status(ot.StatusCode.ERROR, str(e)))
            raise
        finally:
            if output is not None:
                s.set_attribute("output.value", _attr(output))


class _SpanHandle:
    def __init__(self, s):
        self._s = s

    def set(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            self._s.set_attribute(k, _attr(v))

    def output(self, value: Any) -> None:
        self._s.set_attribute("output.value", _attr(value))

    @property
    def trace_id(self) -> str:
        return format(self._s.get_span_context().trace_id, "032x")


def trace(name: Optional[str] = None, kind: str = "agent"):
    """Decorator: wraps a function in a span, recording args and return value."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name or fn.__name__, kind=kind, input={"args": args, "kwargs": kwargs}) as s:
                out = fn(*args, **kwargs)
                s.output(out)
                return out

        return wrapper

    return deco


def tool(fn=None, *, name: Optional[str] = None):
    """Decorator for tool functions (kind=tool)."""
    if fn is None:
        return lambda f: tool(f, name=name)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with span(name or fn.__name__, kind="tool", input=kwargs, **{"tool.name": name or fn.__name__}) as s:
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            s.set(**{"tool.latency_ms": (time.perf_counter() - t0) * 1000})
            s.output(out)
            return out

    return wrapper


def current_trace_id() -> Optional[str]:
    s = ot.get_current_span()
    ctx = s.get_span_context()
    if ctx is None or ctx.trace_id == 0:
        return None
    return format(ctx.trace_id, "032x")


def flush(timeout_ms: int = 5000) -> None:
    p = _state["provider"]
    if p is not None:
        p.force_flush(timeout_ms)


def shutdown() -> None:
    p = _state["provider"]
    if p is not None:
        p.shutdown()
        _state["provider"] = None
