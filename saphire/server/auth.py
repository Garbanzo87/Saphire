"""Authentication, RBAC, API keys, JWT sessions, OIDC SSO, usage metering and quotas.

Principals
  * root key      – `SAPHIRE_API_KEY` (bootstrap). Superadmin over every org. Rotate and replace with org API keys.
  * API key       – `x-api-key: sk_saph_...` created per org with a role (viewer|member|admin|owner). Stored hashed.
  * user (JWT)    – `Authorization: Bearer <jwt>` issued after OIDC login (or `/v1/auth/dev-login` when
                    `SAPHIRE_DEV_LOGIN=1`). Org chosen with `x-org: <slug>` (default: first membership).

Roles (cumulative): viewer < member < admin < owner.
  viewer  read everything in the org
  member  + create/ingest: traces, rollouts, scores, datasets, agents, evals, training, experiments
  admin   + promote/deploy, manage API keys, manage members, org settings read
  owner   + org settings write, delete org
"""
from __future__ import annotations

import hashlib
import secrets
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from . import db as D
from .config import settings

current_principal: ContextVar[Optional["Principal"]] = ContextVar("current_principal", default=None)

ROLE_RANK = {"viewer": 0, "member": 1, "admin": 2, "owner": 3, "superadmin": 4}
# minimum role per permission
PERMISSIONS = {
    "read": "viewer",
    "write": "member",
    "deploy": "admin",
    "keys": "admin",
    "members": "admin",
    "org.read": "admin",
    "org.write": "owner",
    "audit": "admin",
    "usage": "admin",
}


@dataclass
class Principal:
    actor_type: str  # root | api_key | user
    actor_id: str
    label: str
    org: Optional[D.Organization]
    role: str  # viewer|member|admin|owner|superadmin
    user: Optional[D.User] = None
    api_key: Optional[D.ApiKey] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return ROLE_RANK.get(self.role, -1) >= ROLE_RANK[role]

    def can(self, permission: str) -> bool:
        return self.has_role(PERMISSIONS[permission])

    @property
    def is_superadmin(self) -> bool:
        return self.role == "superadmin"


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_api_key(db: Session, org: D.Organization, name: str, role: str = "member", created_by: str = "",
                   expires_in_days: Optional[int] = None, allowed_ips: Optional[list[str]] = None) -> tuple[D.ApiKey, str]:
    if role not in ("viewer", "member", "admin", "owner"):
        raise HTTPException(400, "invalid role")
    import ipaddress

    for cidr in allowed_ips or []:
        try:
            ipaddress.ip_network(cidr, strict=False)
        except ValueError as e:
            raise HTTPException(400, f"invalid IP/CIDR {cidr}") from e
    raw = "sk_saph_" + secrets.token_urlsafe(32)
    row = D.ApiKey(id=D.uid("key"), org_id=org.id, name=name, key_hash=hash_key(raw), prefix=raw[:12], role=role, created_by=created_by,
                   allowed_ips=allowed_ips or [], expires_at=(time.time() + expires_in_days * 86400) if expires_in_days else None)
    db.add(row)
    db.commit()
    return row, raw


# ---------------------------------------------------------------------------
# JWT sessions
# ---------------------------------------------------------------------------
def issue_jwt(user: D.User, ttl_s: Optional[int] = None) -> str:
    import jwt

    now = int(time.time())
    payload = {"sub": user.id, "email": user.email, "name": user.name, "iat": now, "exp": now + (ttl_s or settings.jwt_ttl_s),
               "iss": "saphire"}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_jwt(token: str) -> dict[str, Any]:
    import jwt

    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], issuer="saphire")
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"invalid token: {e}") from e


def upsert_user(db: Session, email: str, name: str = "", provider: str = "", subject: str = "") -> D.User:
    u = db.query(D.User).filter_by(email=email.lower()).one_or_none()
    if u is None:
        u = D.User(id=D.uid("user"), email=email.lower(), name=name, sso_provider=provider, sso_subject=subject,
                   is_superadmin=email.lower() in {e.lower() for e in settings.superadmin_emails})
        db.add(u)
    u.last_login_at = time.time()
    if name:
        u.name = name
    if provider:
        u.sso_provider, u.sso_subject = provider, subject
    db.commit()
    return u


def auto_join(db: Session, user: D.User) -> list[D.Membership]:
    """Join orgs whose settings.allowed_email_domains match the user's email domain (SSO just-in-time provisioning)."""
    domain = user.email.split("@", 1)[-1]
    joined = []
    for org in db.query(D.Organization).all():
        domains = [d.lower() for d in (org.settings or {}).get("allowed_email_domains", [])]
        if domain in domains and db.query(D.Membership).filter_by(org_id=org.id, user_id=user.id).one_or_none() is None:
            m = D.Membership(id=D.uid("mem"), org_id=org.id, user_id=user.id, role=(org.settings or {}).get("default_role", "member"))
            db.add(m)
            joined.append(m)
    if joined:
        db.commit()
    return joined


# ---------------------------------------------------------------------------
# Principal resolution
# ---------------------------------------------------------------------------
def _resolve_org(db: Session, x_org: Optional[str], query_org: Optional[str], memberships: list[D.Membership]) -> tuple[Optional[D.Organization], Optional[D.Membership]]:
    slug = x_org or query_org
    if slug:
        org = db.query(D.Organization).filter_by(slug=slug).one_or_none()
        if org is None:
            raise HTTPException(404, f"unknown org '{slug}'")
        m = next((m for m in memberships if m.org_id == org.id), None)
        return org, m
    if memberships:
        m = memberships[0]
        return db.get(D.Organization, m.org_id), m
    return None, None


async def get_principal(request: Request, x_api_key: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None),
                        x_org: Optional[str] = Header(default=None), db: Session = Depends(D.get_db)) -> Principal:
    """Resolve the caller. Declared `async` so the ContextVar it sets is visible to the (thread-pooled) endpoint."""
    if getattr(request.state, "principal", None) is not None:  # already resolved for this request
        current_principal.set(request.state.principal)
        return request.state.principal
    query_org = request.query_params.get("org")
    key = x_api_key or (authorization.split(" ", 1)[1] if authorization and authorization.lower().startswith("apikey ") else None)
    bearer = authorization.split(" ", 1)[1] if authorization and authorization.lower().startswith("bearer ") else None
    principal: Optional[Principal] = None
    if key:
        if settings.api_key and secrets.compare_digest(key, settings.api_key):
            org, _ = _resolve_org(db, x_org, query_org, [])
            principal = Principal("root", "root", "root-key", org or D.ensure_org(db), "superadmin")
        else:
            row = db.query(D.ApiKey).filter_by(key_hash=hash_key(key)).one_or_none()
            if row is None or row.revoked_at or (row.expires_at and row.expires_at < time.time()):
                raise HTTPException(401, "invalid, revoked or expired API key")
            if row.allowed_ips and not _ip_allowed(client_ip(request), row.allowed_ips):
                raise HTTPException(403, "API key not allowed from this IP address")
            row.last_used_at = time.time()
            db.commit()
            org = db.get(D.Organization, row.org_id)
            if (x_org or query_org) and org.slug != (x_org or query_org):
                raise HTTPException(403, "API key belongs to a different org")
            principal = Principal("api_key", row.id, f"{row.name} ({row.prefix}…)", org, row.role, api_key=row)
    elif bearer and not bearer.startswith("sk_saph_"):
        claims = decode_jwt(bearer)
        user = db.get(D.User, claims["sub"])
        if user is None:
            raise HTTPException(401, "unknown user")
        memberships = db.query(D.Membership).filter_by(user_id=user.id).all()
        org, m = _resolve_org(db, x_org, query_org, memberships)
        role = "superadmin" if user.is_superadmin else (m.role if m else None)
        if role is None:
            raise HTTPException(403, "not a member of this org")
        principal = Principal("user", user.id, user.email, org, role, user=user)
    elif bearer:  # api key passed as bearer
        return await get_principal(request, x_api_key=bearer, authorization=None, x_org=x_org, db=db)
    if principal is None:
        raise HTTPException(401, "authentication required (x-api-key or Authorization: Bearer <jwt>)")
    request.state.principal = principal
    current_principal.set(principal)
    _rate_limit(principal)
    return principal


def client_ip(request: Request) -> str:
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _ip_allowed(ip: str, allowed: list[str]) -> bool:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for cidr in allowed:
        try:
            if addr in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def require(permission: str):
    def _dep(p: Principal = Depends(get_principal)) -> Principal:
        if not p.can(permission):
            raise HTTPException(403, f"role '{p.role}' lacks permission '{permission}'")
        return p

    return _dep


def method_permission(request: Request, p: Principal = Depends(get_principal)) -> Principal:
    """Router-level dependency: GET/HEAD need `read`, everything else `write`. Sensitive routes add stricter checks."""
    perm = "read" if request.method in ("GET", "HEAD", "OPTIONS") else "write"
    if not p.can(perm):
        raise HTTPException(403, f"role '{p.role}' lacks permission '{perm}'")
    return p


# ---------------------------------------------------------------------------
# Rate limiting (per org, per process token bucket) and quotas
# ---------------------------------------------------------------------------
_BUCKETS: dict[str, list[float]] = {}


def _rate_limit(p: Principal) -> None:
    if p.org is None:
        return
    rpm = (p.org.quotas or {}).get("requests_per_minute") or settings.rate_limit_rpm
    if not rpm:
        return
    now = time.time()
    b = _BUCKETS.setdefault(p.org.id, [])
    while b and b[0] < now - 60:
        b.pop(0)
    if len(b) >= rpm:
        raise HTTPException(429, f"rate limit exceeded ({rpm} requests/minute for org '{p.org.slug}')")
    b.append(now)


def meter(db: Session, org_id: Optional[str], metric: str, value: float = 1.0) -> None:
    """Atomic per-day counter increment (UPDATE first; INSERT on miss; tolerate concurrent first-insert races)."""
    if not org_id or value == 0:
        return
    from sqlalchemy import update
    from sqlalchemy.exc import IntegrityError

    day = time.strftime("%Y-%m-%d", time.gmtime())
    res = db.execute(update(D.UsageCounter).where(D.UsageCounter.org_id == org_id, D.UsageCounter.metric == metric, D.UsageCounter.day == day)
                     .values(value=D.UsageCounter.value + value))
    if res.rowcount == 0:
        try:
            with db.begin_nested():
                db.add(D.UsageCounter(id=D.uid("use"), org_id=org_id, metric=metric, day=day, value=value))
        except IntegrityError:  # another request created the row concurrently
            db.execute(update(D.UsageCounter).where(D.UsageCounter.org_id == org_id, D.UsageCounter.metric == metric, D.UsageCounter.day == day)
                       .values(value=D.UsageCounter.value + value))
    db.commit()


def usage_summary(db: Session, org_id: str, days: int = 30) -> dict[str, Any]:
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    rows = db.query(D.UsageCounter).filter(D.UsageCounter.org_id == org_id, D.UsageCounter.day >= cutoff).all()
    totals: dict[str, float] = {}
    daily: dict[str, dict[str, float]] = {}
    for r in rows:
        totals[r.metric] = totals.get(r.metric, 0.0) + r.value
        daily.setdefault(r.day, {})[r.metric] = r.value
    return {"days": days, "totals": totals, "daily": dict(sorted(daily.items()))}


PLAN_DEFAULTS = {
    "free": {"rollouts_per_month": 5000, "jobs_per_day": 20, "requests_per_minute": 300, "members": 3},
    "team": {"rollouts_per_month": 200000, "jobs_per_day": 500, "requests_per_minute": 3000, "members": 25},
    "enterprise": {},
}


def effective_quotas(org: D.Organization) -> dict[str, Any]:
    return {**PLAN_DEFAULTS.get(org.plan, {}), **(org.quotas or {})}


def check_quota(db: Session, org: Optional[D.Organization], metric: str, add: float = 1.0) -> None:
    """Raise 402 when a metered quota would be exceeded (metric ∈ rollouts_per_month | jobs_per_day)."""
    if org is None:
        return
    q = effective_quotas(org)
    if metric == "rollouts_per_month" and q.get(metric):
        used = usage_summary(db, org.id, 30)["totals"].get("rollouts", 0.0)
        if used + add > q[metric]:
            _quota_event(org, metric, used, q[metric])
            raise HTTPException(402, f"quota exceeded: {int(used)}/{q[metric]} rollouts in the last 30 days (plan {org.plan})")
    if metric == "jobs_per_day" and q.get(metric):
        used = usage_summary(db, org.id, 1)["totals"].get("jobs", 0.0)
        if used + add > q[metric]:
            _quota_event(org, metric, used, q[metric])
            raise HTTPException(402, f"quota exceeded: {int(used)}/{q[metric]} jobs today (plan {org.plan})")


def _quota_event(org: D.Organization, metric: str, used: float, limit: Any) -> None:
    try:
        from .webhooks import deliver

        deliver(org.id, "quota.exceeded", {"metric": metric, "used": used, "limit": limit, "plan": org.plan})
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(db: Session, principal: Optional[Principal], action: str, resource_type: str = "", resource_id: str = "", method: str = "",
          path: str = "", status_code: int = 0, ip: str = "", details: Optional[dict[str, Any]] = None) -> None:
    if principal is not None:
        principal.extra["audited"] = True  # the request middleware then skips its generic entry
    row = D.AuditLog(id=D.uid("aud"), org_id=principal.org.id if principal and principal.org else None,
                     actor_type=principal.actor_type if principal else "system", actor_id=principal.actor_id if principal else "",
                     actor_label=principal.label if principal else "system", action=action, resource_type=resource_type,
                     resource_id=resource_id, method=method, path=path, status_code=status_code, ip=ip, details=details or {})
    db.add(row)
    db.commit()
