"""Request-level audit + metering hook (called from the HTTP middleware after each authenticated /v1 request).

Every mutating request (POST/PUT/PATCH/DELETE) becomes an `audit_log` row: who (principal), what (action derived
from the route, e.g. `agents.promote`, `evals.create`), on which resource, status code and client IP. Explicit
`audit()` calls in sensitive handlers add richer details (promotion, key creation, member changes).
All requests increment the org's `api_requests` usage counter.
"""
from __future__ import annotations

import re
import time
from typing import Any

from fastapi import Request

from . import db as D
from .auth import Principal, meter

_ID = re.compile(r"^[a-z]+_[0-9a-f]{12,16}$|^[0-9a-f]{32}$")
_NOISY = {"auth.dev_login", "auth.callback"}  # already audited explicitly with details


def action_for(method: str, path: str) -> tuple[str, str, str]:
    """('/v1/agents/agent_abc/promote', POST) -> ('agents.promote', 'agent', 'agent_abc')."""
    parts = [p for p in path.split("/") if p and p != "v1"]
    resource = parts[0] if parts else ""
    rid = next((p for p in parts[1:] if _ID.match(p)), "")
    verbs = [p for p in parts[1:] if not _ID.match(p)]
    if verbs:
        verb = ".".join(verbs)
    else:
        verb = {"POST": "create", "PUT": "update", "PATCH": "update", "DELETE": "delete"}.get(method, method.lower())
    return f"{resource}.{verb}", resource.rstrip("s"), rid


def record_request(principal: Principal, request: Request, status_code: int, t0: float) -> None:
    db = D.session()
    try:
        org_id = principal.org.id if principal.org else None
        meter(db, org_id, "api_requests", 1)
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and not principal.extra.get("audited"):
            action, rtype, rid = action_for(request.method, request.url.path)
            if action in _NOISY:
                return
            details: dict[str, Any] = {"duration_ms": round((time.perf_counter() - t0) * 1000, 1)}
            if request.query_params:
                details["query"] = dict(request.query_params)
            db.add(D.AuditLog(id=D.uid("aud"), org_id=org_id, actor_type=principal.actor_type, actor_id=principal.actor_id,
                              actor_label=principal.label, action=action, resource_type=rtype, resource_id=rid, method=request.method,
                              path=request.url.path, status_code=status_code, ip=request.client.host if request.client else "",
                              details=details))
            db.commit()
    except Exception:  # noqa: BLE001  - auditing must never break a request
        db.rollback()
    finally:
        db.close()
