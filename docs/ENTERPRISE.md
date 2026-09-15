# Enterprise: multi-tenancy, SSO, RBAC, audit, quotas

## Tenancy model
`Organization` → `Project`s → everything else. Every request resolves to a **principal** with an org and a role; cross-org
access is impossible by construction (`get_or_404` and every list query are org-scoped). Projects are unique per org, so two
tenants can both have a `default` project.

## Principals
| kind | header | notes |
|---|---|---|
| root key | `x-api-key: $SAPHIRE_API_KEY` | superadmin over all orgs; bootstrap only — create org keys, then rotate it |
| org API key | `x-api-key: sk_saph_…` | created per org with a role; stored as SHA-256 hash; optional expiry; revocable |
| user session | `Authorization: Bearer <jwt>` | issued after OIDC SSO (or dev-login); org via `x-org: <slug>` (default: first membership) |

## SSO (OIDC)
Set `SAPHIRE_OIDC_ISSUER` (Okta, Entra ID, Google, Auth0, Keycloak…), `SAPHIRE_OIDC_CLIENT_ID`, `SAPHIRE_OIDC_CLIENT_SECRET`,
`SAPHIRE_OIDC_REDIRECT_URL=https://api.example.com/v1/auth/callback`, `SAPHIRE_FRONTEND_URL`, `SAPHIRE_JWT_SECRET`.
Flow: dashboard → `GET /v1/auth/login` → IdP → `/v1/auth/callback` → userinfo → JWT → `https://app/login/#token=…`.
Just-in-time provisioning: users whose email domain is in `org.settings.allowed_email_domains` join that org with
`settings.default_role`. Superadmins: `SAPHIRE_SUPERADMIN_EMAILS`. Local dev: `SAPHIRE_DEV_LOGIN=1` enables
`POST /v1/auth/dev-login` (never enable in production).

## RBAC
Roles are cumulative: `viewer` (read) < `member` (+ create/ingest agents, datasets, evals, training, scores, rollouts,
experiments) < `admin` (+ promote/deploy, API keys, members, audit, usage) < `owner` (+ org settings). Router-level:
GET requires `read`, mutations `write`; sensitive routes require `deploy` / `keys` / `members` / `org.write`.
`GET /v1/auth/me` returns the effective permissions for the UI.

## Audit log
Every mutating request is recorded (`audit_log`: actor, action such as `agents.promote`, resource, method/path, status, IP,
duration); sensitive handlers add details (versions, key prefixes, member emails). `GET /v1/audit` (admin) with filters;
dashboard page `/audit`. Audit rows are retained a year longer than data retention.

## Usage metering & quotas
Per org/day counters: `rollouts`, `spans`, `tokens`, `jobs`, `api_requests` (`GET /v1/orgs/current/usage`). Plans set
defaults (`free`: 5k rollouts/30d, 20 jobs/day, 300 rpm, 3 members; `team`: 200k/500/3000/25; `enterprise`: unlimited) and
`org.quotas` overrides. Exceeding a quota returns **402** with a clear message; per-org rate limiting returns **429**
(`SAPHIRE_RATE_LIMIT_RPM` or `quotas.requests_per_minute`).

## Data retention
`org.settings.retention_days` or `SAPHIRE_RETENTION_DAYS`; `saphire retention` (Helm CronJob included) deletes raw
traces/spans/rollouts/scores older than the window in batches; aggregates (eval runs, metrics, versions, usage) are kept.

## Bootstrap
```bash
saphire create-org acme --plan enterprise --owner-email cto@acme.com --allowed-domain acme.com
# prints a one-time admin API key for the org
```

## Not included (yet)
SCIM provisioning, per-project roles, customer-managed encryption keys, IP allow-lists, SOC 2 evidence collection.
