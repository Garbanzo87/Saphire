# Deployment

## Local
```bash
pip install -e ".[dev,mcp,providers]"
SAPHIRE_INLINE_JOBS=1 saphire serve            # API + jobs in one process, SQLite at ./data/saphire.db
cd frontend && npm install && npm run dev      # dashboard
```
Separate workers: `saphire serve` (no inline jobs) + one or more `saphire worker`.

## Docker Compose
`docker compose up --build` → Postgres, API (:8000), worker, dashboard (:3000). Set `SAPHIRE_API_KEY`, `OPENAI_API_KEY`, ... in `.env`.
Build the API image with `--build-arg EXTRAS="providers,mcp,postgres,train"` to include TRL/PyTorch for weight updates.

## Kubernetes (Helm)
```bash
helm install saphire deploy/helm/saphire -n saphire --create-namespace \
  --set secrets.values.SAPHIRE_DATABASE_URL=postgresql+psycopg://... \
  --set env.SAPHIRE_ARTIFACT_STORE=s3://my-bucket/saphire --set worker.keda.enabled=true
```
The chart deploys the API (HPA on CPU), workers (optional KEDA scaling on pending jobs, optional GPU), the dashboard, a
retention CronJob, and optionally the env/reward server for external RL trainers plus an Ingress. Without an artifact store
it provisions a RWX PVC shared by API and workers. Plain manifests remain in `deploy/k8s/saphire.yaml`.

## Artifact store (multi-node)
`SAPHIRE_ARTIFACT_STORE=s3://bucket/prefix` (boto3; MinIO/R2 via `AWS_ENDPOINT_URL`) or `file:///shared/path`. Training jobs
publish each version directory and rewrite the agent config to remote URIs; every worker resolves them into a local cache on
demand, so no shared volume is required.

## Configuration (env vars, prefix `SAPHIRE_`)
| Var | Default | |
|---|---|---|
| `SAPHIRE_DATABASE_URL` | `sqlite:///./data/saphire.db` | Postgres: `postgresql+psycopg://user:pass@host/db` |
| `SAPHIRE_API_KEY` | `dev-key` | required in `x-api-key` header |
| `SAPHIRE_ARTIFACTS_DIR` | `./artifacts` | routers, exemplars, checkpoints, datasets |
| `SAPHIRE_INLINE_JOBS` | `false` | run jobs inside the API process |
| `SAPHIRE_CORS_ORIGINS` | `["http://localhost:3000", ...]` | JSON list |
| `SAPHIRE_HOST` / `SAPHIRE_PROJECT` | `http://localhost:8000` / `default` | SDK side |
| `SAPHIRE_JWT_SECRET`, `SAPHIRE_OIDC_*`, `SAPHIRE_FRONTEND_URL`, `SAPHIRE_DEV_LOGIN`, `SAPHIRE_SUPERADMIN_EMAILS` | | SSO / sessions — see docs/ENTERPRISE.md |
| `SAPHIRE_ARTIFACT_STORE`, `SAPHIRE_ARTIFACT_CACHE` | unset | S3/file mirror for artifacts across nodes |
| `SAPHIRE_RATE_LIMIT_RPM`, `SAPHIRE_RETENTION_DAYS` | 0 / 0 | per-org rate limit, data retention |
| `SAPHIRE_ENV_SERVER` | unset | trainer side: use the HTTP env/reward server instead of in-process replay |
| `RAY_ADDRESS` | unset | Ray cluster for `backend=ray` rollouts |

## Production checklist
* Postgres + ≥2 workers; TLS in front; configure OIDC SSO, create org API keys and rotate the bootstrap `SAPHIRE_API_KEY`.
* Traces can contain prompts and tool outputs: apply redaction in `saphire.sdk.tracing._attr` or at the ingest endpoint before storing.
* Pin `judge_model` and dataset seeds so eval numbers are comparable across versions; keep the held-out dataset fixed.
