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

## Kubernetes
`kubectl apply -f deploy/k8s/saphire.yaml` (edit the secret and image names). Workers scale horizontally; add a GPU limit to the worker
deployment for `sft/dpo/grpo` jobs. Artifacts need a shared RWX volume between API and workers.

## Configuration (env vars, prefix `SAPHIRE_`)
| Var | Default | |
|---|---|---|
| `SAPHIRE_DATABASE_URL` | `sqlite:///./data/saphire.db` | Postgres: `postgresql+psycopg://user:pass@host/db` |
| `SAPHIRE_API_KEY` | `dev-key` | required in `x-api-key` header |
| `SAPHIRE_ARTIFACTS_DIR` | `./artifacts` | routers, exemplars, checkpoints, datasets |
| `SAPHIRE_INLINE_JOBS` | `false` | run jobs inside the API process |
| `SAPHIRE_CORS_ORIGINS` | `["http://localhost:3000", ...]` | JSON list |
| `SAPHIRE_HOST` / `SAPHIRE_PROJECT` | `http://localhost:8000` / `default` | SDK side |

## Production checklist
* Postgres + ≥2 workers; put the API behind TLS; rotate `SAPHIRE_API_KEY` (single-key auth — add your gateway's SSO in front).
* Traces can contain prompts and tool outputs: apply redaction in `saphire.sdk.tracing._attr` or at the ingest endpoint before storing.
* Pin `judge_model` and dataset seeds so eval numbers are comparable across versions; keep the held-out dataset fixed.
