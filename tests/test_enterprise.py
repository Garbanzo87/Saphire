"""Multi-tenancy, API keys, RBAC, SSO sessions, audit log, usage metering and quotas."""
import pytest

from saphire.sdk import AgentConfig


@pytest.fixture()
def dev_login(monkeypatch):
    from saphire.server import config

    monkeypatch.setattr(config.settings, "dev_login", True)


def _mk_org(api, slug, plan="free", **kw):
    return api.post("/v1/orgs", json={"name": slug.title(), "slug": slug, "plan": plan, **kw}).json()


def _key(api, slug, role="member"):
    r = api.post("/v1/orgs/current/keys", json={"name": f"{role} key", "role": role}, headers={"x-org": slug}).json()
    assert r["key"].startswith("sk_saph_") and "key_hash" not in r
    return r["key"]


def test_root_key_is_superadmin_and_orgs_are_isolated(api):
    a = _mk_org(api, "acme")
    b = _mk_org(api, "globex")
    assert {o["slug"] for o in api.get("/v1/orgs").json()} >= {"default", "acme", "globex"}
    ka, kb = _key(api, "acme"), _key(api, "globex")
    ha, hb = {"x-api-key": ka}, {"x-api-key": kb}
    agent = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()}, headers=ha).json()
    # same project name in the other org is a different project; other org cannot see the agent
    assert api.get("/v1/agents", headers=hb).json() == []
    assert api.get(f"/v1/agents/{agent['id']}", headers=hb).status_code == 404
    assert api.get(f"/v1/agents/{agent['id']}", headers=ha).status_code == 200
    # key cannot switch orgs
    assert api.get("/v1/agents", headers={**ha, "x-org": "globex"}).status_code == 403
    assert a["id"] != b["id"] and api.get("/v1/projects", headers=ha).json()[0]["org_id"] == a["id"]


def test_rbac_roles(api):
    _mk_org(api, "acme")
    viewer, member, admin = (_key(api, "acme", r) for r in ("viewer", "member", "admin"))
    body = {"config": AgentConfig(name="a", model="mock").model_dump()}
    assert api.post("/v1/agents", json=body, headers={"x-api-key": viewer}).status_code == 403
    assert api.get("/v1/agents", headers={"x-api-key": viewer}).status_code == 200
    ag = api.post("/v1/agents", json=body, headers={"x-api-key": member}).json()
    assert api.post(f"/v1/agents/{ag['id']}/promote", headers={"x-api-key": member}).status_code == 403
    assert api.post(f"/v1/agents/{ag['id']}/promote", headers={"x-api-key": admin}).status_code == 200
    assert api.get("/v1/orgs/current/keys", headers={"x-api-key": member}).status_code == 403
    assert api.get("/v1/orgs/current/keys", headers={"x-api-key": admin}).status_code == 200
    # a member cannot mint an admin key
    assert api.post("/v1/orgs/current/keys", json={"name": "x", "role": "admin"}, headers={"x-api-key": member}).status_code == 403
    assert api.patch("/v1/orgs/current", json={"name": "New"}, headers={"x-api-key": admin}).status_code == 403


def test_key_revocation_and_expiry(api):
    _mk_org(api, "acme")
    k = api.post("/v1/orgs/current/keys", json={"name": "temp", "role": "member", "expires_in_days": 1}, headers={"x-org": "acme"}).json()
    assert api.get("/v1/agents", headers={"x-api-key": k["key"]}).status_code == 200
    api.delete(f"/v1/orgs/current/keys/{k['id']}", headers={"x-org": "acme"})
    assert api.get("/v1/agents", headers={"x-api-key": k["key"]}).status_code == 401
    assert api.get("/v1/agents", headers={"x-api-key": "sk_saph_nope"}).status_code == 401


def test_jwt_sessions_and_memberships(api, dev_login):
    _mk_org(api, "acme", settings={"allowed_email_domains": ["acme.com"], "default_role": "viewer"})
    # SSO-style JIT join by email domain
    api.headers.pop("x-api-key")  # user sessions instead of the root key
    tok = api.post("/v1/auth/dev-login", json={"email": "ann@acme.com", "name": "Ann"}).json()["token"]
    me = api.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert me["actor_type"] == "user" and me["org"] == "acme" and me["role"] == "viewer"
    assert api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()},
                    headers={"Authorization": f"Bearer {tok}"}).status_code == 403
    # promote Ann to admin via root, then she can manage members
    api.post("/v1/orgs/current/members", json={"email": "ann@acme.com", "role": "admin"}, headers={"x-org": "acme", "x-api-key": "dev-key"})
    members = api.get("/v1/orgs/current/members", headers={"Authorization": f"Bearer {tok}", "x-org": "acme"}).json()
    assert members[0]["email"] == "ann@acme.com" and members[0]["role"] == "admin"
    # outsider has no membership
    tok2 = api.post("/v1/auth/dev-login", json={"email": "bob@other.com"}).json()["token"]
    assert api.get("/v1/agents", headers={"Authorization": f"Bearer {tok2}", "x-org": "acme"}).status_code == 403
    assert api.get("/v1/auth/validate", params={"token": tok}).json()["claims"]["email"] == "ann@acme.com"
    assert api.get("/v1/agents", headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


def test_audit_log_and_usage_metering(api):
    _mk_org(api, "acme", plan="team")
    k = _key(api, "acme", "admin")
    h = {"x-api-key": k}
    ag = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()}, headers=h).json()
    api.post(f"/v1/agents/{ag['id']}/promote", headers=h)
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "smoke", "n_per_env": 4}, headers=h).json()
    run = api.post("/v1/evals", json={"agent_id": ag["id"], "dataset_id": ds["id"]}, headers=h).json()
    from conftest import wait_job

    wait_job(api, run["job_id"])
    log = api.get("/v1/audit", headers=h).json()
    actions = [i["action"] for i in log["items"]]
    assert "agents.promote" in actions and "evals.create" in actions and "datasets.create" in actions and "keys.create" in actions
    promote = next(i for i in log["items"] if i["action"] == "agents.promote" and i["details"].get("version"))
    assert promote["actor_type"] == "api_key" and promote["resource_id"] == ag["id"]
    assert api.get("/v1/audit", params={"action": "agents"}, headers=h).json()["total"] >= 2
    usage = api.get("/v1/orgs/current/usage", headers=h).json()
    assert usage["totals"]["rollouts"] == 4 and usage["totals"]["jobs"] == 1 and usage["totals"]["api_requests"] >= 5
    assert usage["totals"]["tokens"] > 0 and usage["quotas"]["rollouts_per_month"] == 200000
    # audit is org-scoped
    _mk_org(api, "globex")
    assert api.get("/v1/audit", headers={"x-api-key": _key(api, "globex", "admin")}).json()["total"] <= 1


def test_quotas_and_rate_limit(api, monkeypatch):
    _mk_org(api, "tiny", plan="free", quotas={"rollouts_per_month": 5, "jobs_per_day": 1})
    k = _key(api, "tiny", "admin")
    h = {"x-api-key": k}
    ag = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()}, headers=h).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "smoke", "n_per_env": 8}, headers=h).json()
    r = api.post("/v1/evals", json={"agent_id": ag["id"], "dataset_id": ds["id"]}, headers=h)
    assert r.status_code == 402 and "rollouts" in r.json()["detail"]
    small = api.post("/v1/datasets", json={"name": "s", "suite": "smoke", "n_per_env": 2}, headers=h).json()
    assert api.post("/v1/evals", json={"agent_id": ag["id"], "dataset_id": small["id"]}, headers=h).status_code == 201
    r = api.post("/v1/training", json={"agent_id": ag["id"], "algorithm": "signals", "params": {}}, headers=h)
    assert r.status_code == 402 and "jobs" in r.json()["detail"]
    from saphire.server import auth

    monkeypatch.setattr(auth.settings, "rate_limit_rpm", 3)
    auth._BUCKETS.clear()
    codes = [api.get("/v1/agents", headers=h).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429
    monkeypatch.setattr(auth.settings, "rate_limit_rpm", 0)


def test_legacy_default_org_still_works(api):
    # the bootstrap key with no org header lands in the default org (backwards compatible)
    assert api.get("/v1/orgs/current").json()["slug"] == "default"
    assert api.get("/v1/auth/me").json()["role"] == "superadmin"
