"""Prometheus-style operational metrics (`GET /metrics`, text exposition, no extra dependency).

Counters: saphire_http_requests_total{method,route,status}, saphire_rollouts_persisted_total, saphire_spans_ingested_total,
saphire_jobs_total{type,status}. Histograms: saphire_http_request_seconds{route}. Gauges (sampled on scrape):
saphire_jobs_pending, saphire_jobs_running, saphire_orgs, saphire_agents_deployed.
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_hist_buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(lambda: [0.0] * (len(_hist_buckets) + 2))  # buckets + sum + count

_ID = re.compile(r"/[a-z]+_[0-9a-f]{12,16}(?=/|$)|/[0-9a-f]{32}(?=/|$)")


def route_template(path: str) -> str:
    return _ID.sub("/{id}", path)


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    with _lock:
        _counters[(name, tuple(sorted(labels.items())))] += value


def observe(name: str, value: float, **labels: str) -> None:
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        h = _hist[key]
        for i, b in enumerate(_hist_buckets):
            if value <= b:
                h[i] += 1
        h[-2] += value
        h[-1] += 1


def _fmt_labels(labels: tuple[tuple[str, str], ...], extra: dict[str, str] | None = None) -> str:
    items = list(labels) + list((extra or {}).items())
    return "{" + ",".join(f'{k}="{str(v).replace(chr(34), "")}"' for k, v in items) + "}" if items else ""


def render(gauges: dict[str, float] | None = None) -> str:
    lines = []
    with _lock:
        names = sorted({n for n, _ in _counters})
        for n in names:
            lines.append(f"# TYPE {n} counter")
            for (name, labels), v in sorted(_counters.items()):
                if name == n:
                    lines.append(f"{n}{_fmt_labels(labels)} {v:g}")
        hnames = sorted({n for n, _ in _hist})
        for n in hnames:
            lines.append(f"# TYPE {n} histogram")
            for (name, labels), h in sorted(_hist.items()):
                if name != n:
                    continue
                cum = 0.0
                for i, b in enumerate(_hist_buckets):
                    cum = h[i]
                    lines.append(f"{n}_bucket{_fmt_labels(labels, {'le': str(b)})} {cum:g}")
                lines.append(f"{n}_bucket{_fmt_labels(labels, {'le': '+Inf'})} {h[-1]:g}")
                lines.append(f"{n}_sum{_fmt_labels(labels)} {h[-2]:g}")
                lines.append(f"{n}_count{_fmt_labels(labels)} {h[-1]:g}")
    for k, v in (gauges or {}).items():
        lines.append(f"# TYPE {k} gauge")
        lines.append(f"{k} {v:g}")
    return "\n".join(lines) + "\n"


def sample_gauges() -> dict[str, float]:
    from . import db as D

    db = D.session()
    try:
        return {"saphire_jobs_pending": float(db.query(D.Job).filter_by(status="pending").count()),
                "saphire_jobs_running": float(db.query(D.Job).filter_by(status="running").count()),
                "saphire_orgs": float(db.query(D.Organization).count()),
                "saphire_agents_deployed": float(db.query(D.Agent).filter_by(status="deployed").count())}
    finally:
        db.close()
