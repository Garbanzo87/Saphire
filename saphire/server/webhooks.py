"""Outbound webhooks with HMAC-SHA256 signatures.

Payload: {"event": ..., "org": slug, "timestamp": ..., "data": {...}}. Header `x-saphire-signature: sha256=<hex>` over the raw
body, `x-saphire-event`, `x-saphire-delivery`. Deliveries are recorded (status, error, attempts) and retried once.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from typing import Any, Optional

import httpx

from . import db as D

EVENTS = ["job.succeeded", "job.failed", "gate.decided", "quota.exceeded", "intelligence.alert", "agent.promoted", "test"]


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post(url: str, secret: str, event: str, delivery_id: str, body: bytes, timeout: float = 10.0) -> tuple[int, str]:
    try:
        r = httpx.post(url, content=body, headers={"content-type": "application/json", "x-saphire-signature": sign(secret, body),
                                                   "x-saphire-event": event, "x-saphire-delivery": delivery_id}, timeout=timeout)
        return r.status_code, "" if r.status_code < 300 else r.text[:300]
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"[:300]


def deliver(org_id: Optional[str], event: str, data: dict[str, Any], sync: bool = False) -> list[str]:
    """Fan an event out to every active webhook of the org subscribed to it. Returns delivery ids."""
    if not org_id:
        return []
    db = D.session()
    try:
        org = db.get(D.Organization, org_id)
        hooks = [h for h in db.query(D.Webhook).filter_by(org_id=org_id, active=True) if not h.events or event in h.events]
        if not hooks:
            return []
        payload = {"event": event, "org": org.slug if org else None, "timestamp": time.time(), "data": data}
        body = json.dumps(payload, default=str).encode()
        ids = []
        for h in hooks:
            d = D.WebhookDelivery(id=D.uid("whd"), webhook_id=h.id, event=event, payload=payload)
            db.add(d)
            ids.append(d.id)
        db.commit()
        jobs = [(h.url, h.secret, d_id) for h, d_id in zip(hooks, ids)]
    finally:
        db.close()

    def _run():
        for url, secret, d_id in jobs:
            code, err = _post(url, secret, event, d_id, body)
            attempts = 1
            if code == 0 or code >= 500:
                time.sleep(0.5)
                code, err = _post(url, secret, event, d_id, body)
                attempts = 2
            db2 = D.session()
            try:
                row = db2.get(D.WebhookDelivery, d_id)
                if row is not None:
                    row.status_code, row.error, row.attempts = code, err, attempts
                    db2.commit()
            finally:
                db2.close()

    if sync:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()
    return ids
